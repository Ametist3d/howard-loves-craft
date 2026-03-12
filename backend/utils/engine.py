import os
import logging
import json
import asyncio
from typing import Dict
import random

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest, ChatRequest
from utils.helpers import (
    read_prompt,
    extract_json,
    get_chat_history,
    kv_get,
    build_verdict_guard,
    extract_last_turn_ban,
    build_state_str,
    build_scene_prompt,
    compress_story,
)

logger = logging.getLogger("keeper_ai.engine")

from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
_REQUEST_BODY_PATH = os.path.join(_BACKEND_DIR, "img_gen", "request_body.json")

os.makedirs(SESSIONS_DIR, exist_ok=True)

# In-Memory Session Storage
active_dbs: Dict[str, SessionDB] = {}
_image_results: Dict[str, str | None] = {}

# Initialize RAG
logger.info("Initializing Embeddings (intfloat/multilingual-e5-large)...")
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

try:
    rules_db = Chroma(persist_directory=os.path.join(DATA_DIR, "coc_rules_db"), embedding_function=emb)
    scen_db  = Chroma(persist_directory=os.path.join(DATA_DIR, "coc_scenario_db"), embedding_function=emb)
    logger.info("ChromaDBs loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ChromaDBs: {e}")
    rules_db, scen_db = None, None


# ─────────────────────────────────────────────────────────────────────────────
# Character generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_character_logic(req: CharGenRequest) -> dict:
    llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434", temperature=0.7)
    chain = PromptTemplate.from_template(read_prompt("character_gen.txt")) | llm
    response_text = chain.invoke({
        "language": req.language,
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected"
    })
    return extract_json(response_text)


# ─────────────────────────────────────────────────────────────────────────────
# Session start
# ─────────────────────────────────────────────────────────────────────────────

async def start_session_logic(req: StartSessionRequest) -> dict:
    session_id = "local_session"

    # 1. Format the setting string
    themes_str = ', '.join(req.themes).upper() if req.themes else "STANDARD"
    era_context = req.era_context or "Cosmic Horror — derive era and aesthetics from the scenario atoms."

    if req.scenarioType == 'custom' and req.customPrompt:
        setting_desc = f"Custom: {req.customPrompt[:50]}..."
        query_text = req.customPrompt
    elif req.scenarioType == 'prebuilt' and req.picked_seed:
        setting_desc = f"Prebuilt: {req.picked_seed[:50]}..."
        query_text = req.picked_seed
        era_context = req.era_context or era_context
    elif req.picked_seed:
        setting_desc = f"Themes: {themes_str}"
        query_text = req.picked_seed
    else:
        setting_desc = req.scenarioType
        query_text = "Lovecraftian cosmic horror mystery hook and secrets"

    info = create_session_db_file(SESSIONS_DIR, "Call of Cthulhu", setting_desc)
    db = SessionDB(info.db_path)
    active_dbs[session_id] = db

    logger.info(f"Created new unique session DB: {info.db_path}")

    # 2. Insert Investigators
    for inv in req.investigators:
        chars = inv.get("characteristics", {})
        attrs = inv.get("attributes", {})

        aid = db.upsert_actor(
            kind="PC",
            name=inv.get("name", "Unknown"),
            description=inv.get("occupation", ""),
            hp=attrs.get("HP", {}).get("current", 10),
            mp=attrs.get("MagicPoints", {}).get("current", 10),
            san=attrs.get("Sanity", {}).get("current", 50),
            stats={
                "str": chars.get("STR", 50), "con": chars.get("CON", 50), "dex": chars.get("DEX", 50),
                "int": chars.get("INT", 50), "pow": chars.get("POW", 50), "app": chars.get("APP", 50),
                "siz": chars.get("SIZ", 50), "edu": chars.get("EDU", 50)
            },
            notes=inv.get("background", "")
        )
        for skill in inv.get("skills", []):
            db.set_skill(aid, skill["name"], skill["value"])

        db.log_event("SYS_INIT", {"note": f"Character {inv.get('name')} registered."})

    # 3. PROACTIVE RAG: Pull Scenario Atoms → Synthesize blueprint → Save to DB
    scenario_atoms_text = ""
    blueprint = {}
    blueprint = {}
    if req.scenarioType == 'prebuilt' and req.picked_seed:
        # ── PREBUILT: full document already in picked_seed — use it directly ──
        scenario_atoms_text = (
            "⚠ PREBUILT SCENARIO — FOLLOW THIS EXACTLY.\n"
            "This is the complete, authoritative scenario document. "
            "Every location, NPC, clue, and plot beat is defined here. "
            "DO NOT invent replacements. DO NOT add locations or NPCs not listed. "
            "Run this scenario as written.\n\n"
            + req.picked_seed
        )
        logger.info(f"Prebuilt scenario loaded directly ({len(req.picked_seed)} chars) — skipping RAG and synthesis.")
        db.log_event("SCENARIO_GENERATED", {"type": "prebuilt", "chars": len(req.picked_seed)})

        cur = db.conn.cursor()
        scenario_setting = req.era_context or "Derive from scenario document above"
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (scenario_atoms_text,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", ("PREBUILT",))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (scenario_setting,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (req.picked_seed[:100],))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (req.language,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)", ("{}",))
        db.conn.commit()

    elif scen_db:
        # ── RANDOM / CUSTOM: existing RAG + synthesis pipeline ──
        logger.info(f"Querying Scenario DB for starting atoms using: '{query_text}'")

        # Pull MORE candidates than needed, then randomly sample to break repetition
        scen_docs = scen_db.similarity_search(query_text, k=15)

        top_docs = scen_docs[:3]
        remaining = scen_docs[3:]
        random.shuffle(remaining)
        mixed_docs = top_docs + remaining[:2]

        # Cross-pollinate: pull 1 atom from a DIFFERENT theme to force novelty
        anti_queries = [
            "ancient ritual sacrifice temple priests",
            "arctic expedition ice buried city",
            "hospital patients memories identity",
            "submarine deep ocean pressure hull breach",
            "wartime bunker coded transmissions",
            "suburban neighborhood wrong underneath",
        ]
        cross_query = random.choice(anti_queries)
        cross_docs = scen_db.similarity_search(cross_query, k=3)
        existing_contents = {d.page_content for d in mixed_docs}
        cross_pick = next((d for d in cross_docs if d.page_content not in existing_contents), None)
        if cross_pick:
            mixed_docs.append(cross_pick)

        random.shuffle(mixed_docs)

        atoms = [f"ATOM {i+1}:\n{doc.page_content.strip()}" for i, doc in enumerate(mixed_docs)]
        raw_atoms_text = "\n\n".join(atoms)

        # Synthesis step: build a unique scenario blueprint from atoms
        logger.info("Synthesizing unique scenario blueprint from atoms...")
        lang = req.language if hasattr(req, 'language') else 'ru'
        try:
            synth_llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434", temperature=0.9)
            synth_chain = PromptTemplate.from_template(read_prompt("scenario_gen.txt")) | synth_llm
            synth_raw = synth_chain.invoke({
                "themes": themes_str,
                "era_context": era_context,
                "language": lang,
                "atoms": raw_atoms_text
            })
            blueprint = extract_json(synth_raw)
            scenario_atoms_text = (
                f"SCENARIO TITLE: {blueprint.get('title', 'Unknown')}\n"
                f"SETTING: {blueprint.get('era_and_setting', '')}\n"
                f"HOOK: {blueprint.get('inciting_hook', '')}\n"
                f"CORE MYSTERY: {blueprint.get('core_mystery', '')}\n"
                f"KEY NPC: {blueprint.get('key_npc', '')}\n"
                f"HIDDEN THREAT: {blueprint.get('hidden_threat', '')}\n"
                f"PLOT TWISTS: {' | '.join(blueprint.get('plot_twists', []))}\n"
                f"ATMOSPHERE: {blueprint.get('atmosphere_notes', '')}"
            )
            logger.info(f"Scenario blueprint synthesized: {blueprint.get('title', '?')}")
        except Exception as e:
            logger.warning(f"Scenario synthesis failed ({e}), falling back to raw atoms.")
            scenario_atoms_text = raw_atoms_text

        db.log_event("SCENARIO_GENERATED", {"query": query_text})

        # Populate DB tables from blueprint skeleton
        if isinstance(blueprint, dict):
            loc_id_map = {}
            for loc in blueprint.get("locations", []):
                lid = db.upsert_location(
                    name=loc.get("name", "Unknown"),
                    description=loc.get("description", ""),
                    tags=loc.get("tags", "")
                )
                loc_id_map[loc.get("name", "")] = lid

            for npc in blueprint.get("npcs", []):
                role = npc.get("role", "neutral")
                kind = "ENEMY" if "enemy" in role else "NPC"
                db.upsert_actor(
                    kind=kind,
                    name=npc.get("name", "Unknown"),
                    description=npc.get("description", ""),
                    notes=npc.get("secret", "")
                )

            for clue in blueprint.get("clues", []):
                loc_name = clue.get("location", "")
                db.upsert_clue(
                    title=clue.get("title", "Clue"),
                    content=clue.get("content", ""),
                    status="hidden",
                    location_id=loc_id_map.get(loc_name)
                )

            for thread in blueprint.get("plot_threads", []):
                db.upsert_thread(
                    name=thread.get("name", "Thread"),
                    stakes=thread.get("stakes", ""),
                    max_progress=thread.get("steps", 4)
                )

        scenario_setting = (
            blueprint.get('era_and_setting') or era_context
            if isinstance(blueprint, dict)
            else era_context
        )
        cur = db.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (scenario_atoms_text,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (themes_str,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (era_context,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (req.picked_seed[:100],))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (req.language,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)",
                    (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",))
        db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", era_context))
        db.conn.commit()

        # DEBUG
        print(f"==>> [SESSION START] era_context passed in : {repr(req.era_context)}")
        print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
        print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
        print(f"==>> [SESSION START] themes_str            : {repr(themes_str)}")

    # 4. Trigger opening narration
    start_msg = (
        "Начни историю. Опиши стартовую локацию."
        if req.language == 'ru'
        else "Start the story. Describe the starting location."
    )
    return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))


# ─────────────────────────────────────────────────────────────────────────────
# Avatar generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_avatar_logic(req) -> dict:
    tpl = PromptTemplate.from_template(
        "You are a prompt engineer for Flux, a natural-language image model.\n"
        "Write a portrait prompt for this character. Era/setting: {era}.\n\n"
        "Character: {occupation}, {description}\n\n"
        "RULES:\n"
        "- English only\n"
        "- 1-2 sentences describing: face and expression, clothing appropriate to the era, posture/mood\n"
        "- Framing: upper body portrait, neutral or slightly dramatic background\n"
        "- No character names\n"
        "- End with exactly: 'Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.'\n"
        "- Output ONLY the prompt, nothing else\n\n"
        "PORTRAIT PROMPT:"
    )
    llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434", temperature=0.3)
    portrait_prompt = (tpl | llm).invoke({
        "era": req.era_context or "1920s Lovecraftian Horror",
        "occupation": req.occupation,
        "description": req.physical_description
    }).strip()
    portrait_prompt = " ".join(line.strip() for line in portrait_prompt.splitlines() if line.strip())

    logger.info(f"Avatar prompt for {req.name}: {portrait_prompt}")

    try:
        from img_gen.comfy_client import ComfyClient
        _comfy = ComfyClient(_COMFY_BASE_URL)
        with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as f:
            body = json.load(f)
        body["params"]["prompt"] = portrait_prompt
        body["params"]["width"] = 480
        body["params"]["height"] = 640
        img_result = _comfy.generate(body)
        return {"image_url": img_result["image_url"], "portrait_prompt": portrait_prompt}
    except Exception as e:
        logger.warning(f"Avatar generation failed for {req.name}: {e}")
        return {"image_url": None, "portrait_prompt": portrait_prompt}


# ─────────────────────────────────────────────────────────────────────────────
# Main chat handler
# ─────────────────────────────────────────────────────────────────────────────

async def handle_chat_logic(req: ChatRequest) -> dict:
    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    db.log_event("CHAT", {"role": "User", "content": req.message})

    cur = db.conn.cursor()

    # 1. Load kv_store session values
    campaign_atoms  = kv_get(cur, "scenario_atoms")
    themes          = kv_get(cur, "scenario_themes", "STANDARD")
    setting_override = kv_get(cur, "scenario_setting", "Lovecraftian Horror Lore")
    era_override    = kv_get(cur, "era_context", "1920s Lovecraftian Horror")
    session_language = kv_get(cur, "language", "en")

    # DEBUG
    print(f"==>> [CHAT] setting_override read from DB : {repr(setting_override)}")
    print(f"==>> [CHAT] era_override read from DB     : {repr(era_override)}")

    # 2. Build context string
    context_str = ""

    # Story digest — primary long-term continuity anchor
    story_digest = kv_get(cur, "story_digest")
    if story_digest:
        context_str += (
            "═══════════════════════════════════════════\n"
            "STORY SO FAR — READ THIS FIRST\n"
            "This is a factual record of everything that has happened in this session.\n"
            "Treat it as ground truth. Do NOT contradict, repeat, or re-discover anything listed here.\n"
            f"{story_digest}\n"
            "═══════════════════════════════════════════\n\n"
        )

    # Scenario blueprint
    if campaign_atoms:
        context_str += (
            f"--- SCENARIO BLUEPRINT (THEMES: {themes}) ---\n"
            "You are running THIS specific scenario. Every narrative choice must serve it.\n"
            "DO NOT fall back to generic 1920s tropes. USE the setting, hook, NPCs, and threat below.\n"
            f"{campaign_atoms}\n"
            "---------------------------------------------\n\n"
        )

    # 3. Standard RAG (skip on the first "Start the story" to avoid polluting context)
    is_start_msg = "Начни историю" in req.message or "Start the story" in req.message
    if req.rag_enabled and rules_db and scen_db and not is_start_msg:
        all_docs = (
            rules_db.similarity_search(req.message, k=req.top_k)
            + scen_db.similarity_search(req.message, k=req.top_k)
        )
        context_str += "--- RELEVANT RULEBOOK/SCENARIO LORE ---\n"
        context_str += "\n\n".join([doc.page_content for doc in all_docs])

    # 4. State injection
    pack = db.build_prompt_state_pack()
    state_str = build_state_str(pack)

    # 5. Per-turn guards
    verdict_guard = build_verdict_guard(req.message)
    last_turn_ban = extract_last_turn_ban(db)

    # 6. Invoke LLM
    llm = OllamaLLM(
        model="gemma3:27b",
        base_url="http://localhost:11434",
        temperature=req.temperature,
        num_ctx=req.num_ctx,
    )
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt")) | llm

    response_text = chain.invoke({
        "language": session_language,
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str + verdict_guard,
        "era_context": setting_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": last_turn_ban,
    })

    result = extract_json(response_text)

    # 7. Background image generation
    gen_id = __import__("uuid").uuid4().hex
    _image_results[gen_id] = "pending"
    result["generation_id"] = gen_id
    result["image_url"] = None

    async def _generate_image_bg(gid: str, narrative: str, setting: str, era: str):
        try:
            from img_gen.comfy_client import ComfyClient
            _comfy = ComfyClient(_COMFY_BASE_URL)

            visual_history = ""
            char_visuals = ""
            if req.session_id in active_dbs:
                _db = active_dbs[req.session_id]
                _cur = _db.conn.cursor()
                visual_history = kv_get(_cur, "visual_history")
                _cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'")
                rows = _cur.fetchall()
                if rows:
                    char_visuals = "ESTABLISHED CHARACTERS (maintain consistent appearance):\n"
                    char_visuals += "\n".join(f"- {r['value']}" for r in rows)

            scene_prompt = build_scene_prompt(
                narrative, era=era, setting=setting,
                visual_history=visual_history,
                char_visuals=char_visuals,
            )
            logger.info(f"Image prompt [{gid}]: {scene_prompt}")
            with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as f:
                body = json.load(f)
            body["params"]["prompt"] = scene_prompt
            img_result = _comfy.generate(body)
            _image_results[gid] = img_result["image_url"]
            logger.info(f"Image ready [{gid}]: {img_result['image_url']}")
        except Exception as e:
            logger.warning(f"Image generation failed [{gid}]: {e}")
            _image_results[gid] = None

    asyncio.create_task(_generate_image_bg(
        gen_id,
        result.get("narrative", ""),
        setting=setting_override,
        era=era_override,
    ))

    # 8. Log Keeper turn + trigger compression every 5 turns
    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})

    turn_count = int(kv_get(cur, "turn_count", "0")) + 1
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)", (str(turn_count),))
    db.conn.commit()

    if turn_count % 5 == 0:
        asyncio.create_task(compress_story(db))
        logger.info(f"Story compression triggered at turn {turn_count}")

    # 9. Apply state updates
    if state_updates := result.get("state_updates"):

        if target_name := state_updates.get("character_name"):
            all_actors = db.list_actors("PC") + db.list_actors("NPC") + db.list_actors("ENEMY")
            target = next((a for a in all_actors if target_name.lower() in a["name"].lower()), None)
            if target:
                hp_change  = int(state_updates.get("hp_change", 0) or 0)
                san_change = int(state_updates.get("sanity_change", 0) or 0)
                mp_change  = int(state_updates.get("mp_change", 0) or 0)

                if hp_change != 0 or san_change != 0 or mp_change != 0:
                    new_hp  = max(0, (target.get("hp")  or 0) + hp_change)
                    new_san = max(0, (target.get("san") or 0) + san_change)
                    new_mp  = max(0, (target.get("mp")  or 0) + mp_change)

                    status = target.get("status", "ok")
                    if new_hp == 0:       status = "dead"
                    elif new_san == 0:    status = "insane"
                    elif hp_change < 0:   status = "injured"

                    db.upsert_actor(
                        actor_id=target["id"], kind=target["kind"], name=target["name"],
                        hp=new_hp, san=new_san, mp=new_mp, status=status
                    )
                    db.log_event("STATE_UPDATE", {
                        "actor": target["name"],
                        **({f"hp": f"{hp_change:+} → {new_hp}"} if hp_change else {}),
                        **({f"san": f"{san_change:+} → {new_san}"} if san_change else {}),
                        **({f"mp": f"{mp_change:+} → {new_mp}"} if mp_change else {}),
                        "status": status,
                    })

                if item_add := state_updates.get("inventory_add", ""):
                    db.upsert_clue(title=item_add, content=f"Carried by {target['name']}", status="found")
                    db.log_event("INVENTORY", {"actor": target["name"], "added": item_add})

                if item_remove := state_updates.get("inventory_remove", ""):
                    db.log_event("INVENTORY", {"actor": target["name"], "removed": item_remove})

        if new_location := state_updates.get("location_name", ""):
            existing = db.get_location_by_name(new_location)
            loc_id = existing["id"] if existing else db.upsert_location(
                name=new_location,
                description=state_updates.get("location_description", "")
            )
            for pc in db.list_actors("PC"):
                db.upsert_actor(actor_id=pc["id"], kind="PC", name=pc["name"], location_id=loc_id)
            db.log_event("LOCATION_CHANGE", {"location": new_location})

        if clue_found := state_updates.get("clue_found", ""):
            clues = db.list_clues(status="hidden")
            match = next((c for c in clues if clue_found.lower() in c["title"].lower()), None)
            if match:
                db.upsert_clue(clue_id=match["id"], title=match["title"],
                               content=match["content"], status="found")
            else:
                db.upsert_clue(title=clue_found,
                               content=state_updates.get("clue_content", ""), status="found")
            db.log_event("CLUE_FOUND", {"clue": clue_found})

        if thread_name := state_updates.get("thread_progress", ""):
            threads = db.list_threads()
            match = next((t for t in threads if thread_name.lower() in t["name"].lower()), None)
            if match:
                db.upsert_thread(
                    thread_id=match["id"], name=match["name"],
                    progress=min(match["progress"] + 1, match["max_progress"]),
                    max_progress=match["max_progress"], stakes=match.get("stakes", "")
                )
                db.log_event("THREAD_PROGRESS", {"thread": match["name"],
                                                   "progress": match["progress"] + 1})

    return result
