import os
import logging
import json
from typing import Dict
import random

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest, ChatRequest
from utils.helpers import (
    read_prompt, extract_json, get_chat_history, get_llm, compress_story,
    build_scene_prompt, generate_avatar_logic, apply_state_updates,
    normalize_language_code, get_language_name, assemble_keeper_prompt,
    infer_scene_stall_level, build_verdict_guard, extract_last_turn_ban,
    has_roll_verdict, build_scene_loop_guard
)

logger = logging.getLogger("keeper_ai.engine")

from utils.prompt_translate import ensure_translated_prompts

from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL


# _COMFY_BASE_URL = "https://jacksonville-arms-latest-johnson.trycloudflare.com"  # update as needed
# _COMFY_BASE_URL = "https://provided-feeds-pipe-avatar.trycloudflare.com"  # update as needed

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

import datetime
_DEBUG_LOG = os.path.join(DATA_DIR, "prebuilt_debug.log")

def _dbg(tag: str, content: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n[{ts}] {tag}\n{'='*60}\n{content}\n")

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


async def generate_character_logic(req: CharGenRequest) -> dict:
    language = normalize_language_code(req.language)
    prompt_dir = ensure_translated_prompts("global_chargen", language)

    llm = get_llm(temperature=0.7)
    chain = PromptTemplate.from_template(
        read_prompt("character_gen.txt", prompt_dir=prompt_dir)
    ) | llm

    response_text = chain.invoke({
        "language": language,
        "language_name": get_language_name(language),
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected",
    })

    
    return extract_json(response_text)

async def start_session_logic(req: StartSessionRequest) -> dict:
    session_id = "local_session"

    lang = normalize_language_code(req.language)
    prompt_dir = ensure_translated_prompts(session_id, lang)
    
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

    # 3. Build scenario blueprint
    scenario_atoms_text = ""
    blueprint = {}
    lang = normalize_language_code(lang)

    if req.scenarioType == 'prebuilt' and req.picked_seed:
        # ── PREBUILT PATH ──────────────────────────────────────────────────────
        logger.info("Prebuilt scenario selected — extracting blueprint from source content.")
        scenario_source_text = req.picked_seed

        extract_prompt = (
            "You are a Call of Cthulhu scenario analyst. "
            "Below is the full text of a published scenario. "
            "Extract its structure into JSON. Do NOT invent anything — only use what is in the text.\n\n"
            "SCENARIO TEXT:\n"
            f"{scenario_source_text[:6000]}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  title (str), era_and_setting (str), inciting_hook (str), core_mystery (str),\n"
            "  key_npc (str), hidden_threat (str), atmosphere_notes (str),\n"
            "  plot_twists (list[str]),\n"
            "  locations (list of {name, description, tags}),\n"
            "  npcs (list of {name, description, role, secret}),\n"
            "  clues (list of {title, content, location}),\n"
            "  plot_threads (list of {name, stakes, steps})\n"
            "JSON:"
        )
        try:
            extract_llm = get_llm(temperature=0.1)
            extract_raw = extract_llm.invoke(extract_prompt)
            _dbg("EXTRACT PROMPT", extract_prompt[:2000])
            _dbg("EXTRACT RAW RESPONSE", str(extract_raw))
            logger.info(f"[PREBUILT] Raw extract response (first 300): {str(extract_raw)[:300]}")
            blueprint = extract_json(extract_raw)
            logger.info(f"[PREBUILT] blueprint keys={list(blueprint.keys()) if isinstance(blueprint, dict) else 'NOT A DICT'}")

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
            logger.info(f"Prebuilt blueprint extracted: {blueprint.get('title', '?')}")

            if lang != "en":
                language_name = get_language_name(lang)
                logger.info(f"[PREBUILT] Translating atoms to {language_name}...")
                translate_prompt = (
                    "Translate the following text into {language_name}. "
                    "Do NOT translate proper nouns (person names, place names, ship names, artifact names). "
                    "Output ONLY the translated text, nothing else:\n\n"
                    f"{scenario_atoms_text}"
                )
                try:
                    translated_atoms = get_llm(temperature=0.1).invoke(translate_prompt).strip()
                    _dbg("TRANSLATE PROMPT", translate_prompt)
                    _dbg("TRANSLATE RAW RESPONSE", translated_atoms)
                    if translated_atoms:
                        scenario_atoms_text = translated_atoms
                    logger.info(f"[PREBUILT] scenario_atoms_text after translation: {scenario_atoms_text[:200]}")
                except Exception as te:
                    logger.warning(f"Atoms translation failed ({te}), keeping English.")

                for loc in blueprint.get('locations', []):
                    if loc.get('description'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{loc['description']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD location.description [{loc['name']}]", f"IN: {loc['description']}\nOUT: {raw}")
                        loc['description'] = raw or loc['description']

                for npc in blueprint.get('npcs', []):
                    for field in ('description', 'secret'):
                        if npc.get(field):
                            field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{npc[field]}"
                            raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                            _dbg(f"FIELD npc.{field} [{npc['name']}]", f"IN: {npc[field]}\nOUT: {raw}")
                            npc[field] = raw or npc[field]

                for i, clue in enumerate(blueprint.get('clues', [])):
                    if clue.get('content'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{clue['content']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD clue.content [{clue.get('title', i)}]", f"IN: {clue['content']}\nOUT: {raw}")
                        clue['content'] = raw or clue['content']

                for thread in blueprint.get('plot_threads', []):
                    if thread.get('stakes'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{thread['stakes']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD thread.stakes [{thread['name']}]", f"IN: {thread['stakes']}\nOUT: {raw}")
                        thread['stakes'] = raw or thread['stakes']

        except Exception as e:
            logger.warning(f"Prebuilt extraction failed ({e}), using raw scenario text.")
            blueprint = {}
            scenario_atoms_text = scenario_source_text[:4000]

    elif scen_db:
        # ── RANDOM / CUSTOM PATH ───────────────────────────────────────────────
        logger.info(f"Querying Scenario DB for starting atoms using: '{query_text}'")
        scen_docs = scen_db.similarity_search(query_text, k=15)
        top_docs = scen_docs[:3]
        remaining = scen_docs[3:]
        random.shuffle(remaining)
        mixed_docs = top_docs + remaining[:2]

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

        logger.info("Synthesizing unique scenario blueprint from atoms...")
        try:
            synth_llm = get_llm(temperature=0.9)
            synth_chain = PromptTemplate.from_template(
                read_prompt("scenario_gen.txt", prompt_dir=prompt_dir)
            ) | synth_llm

            synth_raw = synth_chain.invoke({
                "themes": themes_str,
                "era_context": era_context,
                "language": lang,
                "language_name": get_language_name(lang),
                "atoms": raw_atoms_text,
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
            blueprint = {}
            scenario_atoms_text = raw_atoms_text

    # ── SHARED: save everything to DB — runs for BOTH paths ───────────────────
    _dbg("FINAL scenario_atoms_text", scenario_atoms_text)
    _dbg("FINAL blueprint (first 1000)", json.dumps(blueprint, ensure_ascii=False)[:1000])
    logger.info(f"[SESSION] Final scenario_atoms_text (first 300): {scenario_atoms_text[:300]}")
    logger.info(f"[SESSION] language being stored: {repr(lang)}")

    db.log_event("SCENARIO_GENERATED", {
        "query": query_text if req.scenarioType != 'prebuilt' else blueprint.get('title', 'prebuilt')
    })

    if isinstance(blueprint, dict) and blueprint:
        loc_id_map = {}
        for loc in blueprint.get("locations", []):
            tags_raw = loc.get("tags", "")
            tags_str = ", ".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw or "")
            lid = db.upsert_location(
                name=str(loc.get("name", "Unknown")),
                description=str(loc.get("description", "")),
                tags=tags_str
            )
            loc_id_map[loc.get("name", "")] = lid

        for npc in blueprint.get("npcs", []):
            role = str(npc.get("role", "neutral"))
            kind = "ENEMY" if "enemy" in role else "NPC"
            db.upsert_actor(
                kind=kind,
                name=str(npc.get("name", "Unknown")),
                description=str(npc.get("description", "")),
                notes=str(npc.get("secret", ""))
            )

        for clue in blueprint.get("clues", []):
            loc_name = clue.get("location", "")
            db.upsert_clue(
                title=str(clue.get("title", "Clue")),
                content=str(clue.get("content", "")),
                status="hidden",
                location_id=loc_id_map.get(loc_name)
            )

        for thread in blueprint.get("plot_threads", []):
            steps_raw = thread.get("steps", 4)
            steps = int(steps_raw) if isinstance(steps_raw, (int, float, str)) else 4
            db.upsert_thread(
                name=str(thread.get("name", "Thread")),
                stakes=str(thread.get("stakes", "")),
                max_progress=steps
            )

    scenario_setting = str(
        blueprint.get('era_and_setting') or era_context
        if isinstance(blueprint, dict)
        else era_context
    )
    cur = db.conn.cursor()
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (str(scenario_atoms_text),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (str(themes_str),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (str(era_context),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (str(req.picked_seed[:100]) if req.picked_seed else "",))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (lang,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_dir', ?)", (prompt_dir,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)",
        (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",))
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", str(era_context)))
    db.conn.commit()

    print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
    print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
    print(f"==>> [SESSION START] language stored        : {repr(lang)}")

    # 4. Trigger opening narration
    start_msg = "Start the story. Open with the first playable scene and describe the starting location."
    return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))
    
async def handle_chat_logic(req: ChatRequest) -> dict:
    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    db.log_event("CHAT", {"role": "User", "content": req.message})
    dead_pcs = [a for a in db.list_actors("PC") if a.get("status") in ("dead", "insane")]
    if dead_pcs:
        names = ", ".join(a["name"] for a in dead_pcs)
        req = req.model_copy(update={"message": 
            req.message + f"\n\n[KEEPER NOTE: {names} are dead/incapacitated and cannot act. "
            f"Do not narrate their actions. Acknowledge their fate if relevant.]"
        })
    
    # 1. Fetch Permanent Campaign Module (Atoms) from DB
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_atoms'")
    row_atoms = cur.fetchone()
    campaign_atoms = row_atoms["value"] if row_atoms else ""
    
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_themes'")
    row_themes = cur.fetchone()
    themes = row_themes["value"] if row_themes else "STANDARD"

    # 2. Build Context
    context_str = ""
    
    # Inject rolling story digest — the model's primary continuity anchor
    cur.execute("SELECT value FROM kv_store WHERE key='story_digest'")
    row_digest = cur.fetchone()
    if row_digest and row_digest["value"]:
        context_str += f"""
═══════════════════════════════════════════
STORY SO FAR — READ THIS FIRST
This is a factual record of everything that has happened in this session.
Treat it as ground truth. Do NOT contradict, repeat, or re-discover anything listed here.
{row_digest["value"]}
═══════════════════════════════════════════\n\n"""

    if campaign_atoms:
        context_str += f"""
--- SCENARIO BLUEPRINT (THEMES: {themes}) ---
You are running THIS specific scenario. Every narrative choice must serve it.
DO NOT fall back to generic 1920s tropes. USE the setting, hook, NPCs, and threat below.
{campaign_atoms}
---------------------------------------------\n\n"""

    # 3. Standard RAG (Skip standard RAG on the very first "Start the story" message to avoid polluting context)
    req_lower = req.message.lower()
    is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
    if req.rag_enabled and rules_db and scen_db and not is_start_msg:
        all_docs = rules_db.similarity_search(req.message, k=req.top_k) + scen_db.similarity_search(req.message, k=req.top_k)
        context_str += "--- RELEVANT RULEBOOK/SCENARIO LORE ---\n"
        context_str += "\n\n".join([doc.page_content for doc in all_docs])
    
    # 4. State Injection
        # 4. State Injection — full scenario skeleton
    pack = db.build_prompt_state_pack()
    # Only surface NPCs the Keeper has already narrated — prevents blueprint leaking names
    all_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=60)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ).lower()
    met_npcs = [
        line for line in (pack.get('npcs_text') or '').splitlines()
        if any(word.lower() in all_chat for word in line.split() if len(word) > 4)
    ] or ["(none yet)"]
    state_str = (
        f"INVESTIGATORS:\n{pack.get('investigators_text')}\n\n"
        f"MET NPCs (only these have been introduced — DO NOT reference others):\n"
        + "\n".join(met_npcs) + "\n\n"
        f"CURRENT LOCATION:\n{pack.get('location_text')}\n\n"
        f"PLOT THREADS:\n{pack.get('threads_text')}\n\n"
        f"DISCOVERED CLUES:\n{pack.get('clues_text')}"
    )

    # 5. Invoke LLM via modular Keeper prompt stack
    llm = get_llm(temperature=req.temperature)

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row_setting = cur.fetchone()
    setting_override = row_setting["value"] if row_setting else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row_era = cur.fetchone()
    era_override = row_era["value"] if row_era else "1920s Lovecraftian Horror"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row_lang = cur.fetchone()
    session_language = normalize_language_code(row_lang["value"] if row_lang else "en")

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row_prompt_dir = cur.fetchone()
    prompt_dir = row_prompt_dir["value"] if row_prompt_dir else None

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=has_roll_verdict(req.message),
        include_scene_progression=infer_scene_stall_level(db) >= 1,
        include_opening_scene=is_start_msg,
        prompt_dir=prompt_dir,
    )

    if '"narrative"' in keeper_system_prompt:
        idx = keeper_system_prompt.find('"narrative"')
        start = max(0, idx - 200)
        end = min(len(keeper_system_prompt), idx + 400)
        logger.info("handle_chat_logic(): prompt around narrative:\n%s", keeper_system_prompt[start:end])

    chain = PromptTemplate.from_template(keeper_system_prompt) | llm

    scene_loop_guard = build_scene_loop_guard(db)

    response_text = chain.invoke({
        "language": session_language,
        "language_name": get_language_name(session_language),
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str + "\n\n" + scene_loop_guard,
        "era_context": setting_override + " " + era_override, 
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": extract_last_turn_ban(db),
    })

    logger.info("handle_chat_logic(): raw LLM response preview: %r", str(response_text)[:4000])
    result = extract_json(response_text)

    # ── IMAGE GENERATION ──────────────────────────────────────────────────────────
    import json as _json
    from pathlib import Path as _Path

    
    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    import asyncio, uuid as _uuid
    gen_id = _uuid.uuid4().hex
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
                db = active_dbs[req.session_id]
                cur = db.conn.cursor()
                cur.execute("SELECT value FROM kv_store WHERE key='visual_history'")
                row = cur.fetchone()
                visual_history = row["value"] if row else ""

                # Load all stored character visual descriptions
                cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'")
                rows = cur.fetchall()
                if rows:
                    char_visuals = "ESTABLISHED CHARACTERS (maintain consistent appearance):\n"
                    char_visuals += "\n".join(f"- {r['value']}" for r in rows)

            scene_prompt = build_scene_prompt(
                narrative, era=era, setting=setting,
                visual_history=visual_history,
                char_visuals=char_visuals          # ← NEW
            )
            logger.info(f"Image prompt [{gid}]: {scene_prompt}")
            with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as _f:
                _body = json.load(_f)
            _body["params"]["prompt"] = scene_prompt
            img_result = _comfy.generate(_body)
            _image_results[gid] = img_result["image_url"]
            logger.info(f"Image ready [{gid}]: {img_result['image_url']}")
        except Exception as _e:
            logger.warning(f"Image generation failed [{gid}]: {_e}")
            _image_results[gid] = None

    asyncio.create_task(_generate_image_bg(
        gen_id,
        result.get("narrative", ""),
        setting=setting_override,
        era=era_override            
    ))
    # ─────────────────────────────────────────────────────────────────────────────
    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})
    
    # Increment turn counter and compress every 5 turns
    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row_tc = cur.fetchone()
    turn_count = int(row_tc["value"]) + 1 if row_tc else 1
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)", (str(turn_count),))
    db.conn.commit()
    if turn_count % 5 == 0:
        import asyncio
        asyncio.create_task(compress_story(db))
        logger.info(f"Story compression triggered at turn {turn_count}")
    
    # 6. Apply State Updates
    apply_state_updates(db, result)
    return result


async def stream_chat_logic(req: ChatRequest):
    """
    Streaming version of handle_chat_logic.
    Yields SSE lines:
      data: {"type":"token","text":"..."}   — one per chunk as LLM generates
      data: {"type":"done","payload":{...}}  — full result once generation ends
    """
    import asyncio, json as _json

    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    db.log_event("CHAT", {"role": "User", "content": req.message})

    cur = db.conn.cursor()

    # ── Build context (identical to handle_chat_logic) ───────────────────────
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_atoms'")
    row = cur.fetchone(); campaign_atoms = row["value"] if row else ""

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_themes'")
    row = cur.fetchone(); themes = row["value"] if row else "STANDARD"

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row = cur.fetchone(); setting_override = row["value"] if row else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row = cur.fetchone(); era_override = row["value"] if row else "1920s Lovecraftian Horror"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row = cur.fetchone(); session_language = normalize_language_code(row["value"] if row else "en")

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row = cur.fetchone(); prompt_dir = row["value"] if row else None

    context_str = ""
    cur.execute("SELECT value FROM kv_store WHERE key='story_digest'")
    row = cur.fetchone()
    if row and row["value"]:
        context_str += (
            "═══════════════════════════════════════════\n"
            "STORY SO FAR — READ THIS FIRST\n"
            "Treat it as ground truth. Do NOT contradict or re-discover anything listed here.\n"
            f"{row['value']}\n"
            "═══════════════════════════════════════════\n\n"
        )

    if campaign_atoms:
        context_str += (
            f"--- SCENARIO BLUEPRINT (THEMES: {themes}) ---\n"
            "You are running THIS specific scenario. USE the setting, hook, NPCs, and threat below.\n"
            f"{campaign_atoms}\n"
            "---------------------------------------------\n\n"
        )

    req_lower = req.message.lower()
    is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
    if req.rag_enabled and rules_db and scen_db and not is_start_msg:
        all_docs = (rules_db.similarity_search(req.message, k=req.top_k)
                    + scen_db.similarity_search(req.message, k=req.top_k))
        context_str += "--- RELEVANT RULEBOOK/SCENARIO LORE ---\n"
        context_str += "\n\n".join([doc.page_content for doc in all_docs])

    pack = db.build_prompt_state_pack()
    # Only surface NPCs the Keeper has already narrated — prevents blueprint leaking names
    all_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=60)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ).lower()
    met_npcs = [
        line for line in (pack.get('npcs_text') or '').splitlines()
        if any(word.lower() in all_chat for word in line.split() if len(word) > 4)
    ] or ["(none yet)"]
    state_str = (
        f"INVESTIGATORS:\n{pack.get('investigators_text')}\n\n"
        f"MET NPCs (only these have been introduced — DO NOT reference others):\n"
        + "\n".join(met_npcs) + "\n\n"
        f"CURRENT LOCATION:\n{pack.get('location_text')}\n\n"
        f"PLOT THREADS:\n{pack.get('threads_text')}\n\n"
        f"DISCOVERED CLUES:\n{pack.get('clues_text')}"
    )

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=has_roll_verdict(req.message),
        include_scene_progression=infer_scene_stall_level(db) >= 1,
        include_opening_scene=is_start_msg,
        prompt_dir=prompt_dir,
    )

    # ── Stream tokens from LLM ───────────────────────────────────────────────
    llm = get_llm(temperature=req.temperature)
    chain = PromptTemplate.from_template(keeper_system_prompt) | llm
    verdict_guard = build_verdict_guard(req.message)
    scene_loop_guard = build_scene_loop_guard(db)

    prompt_vars = {
        "language": session_language,
        "language_name": get_language_name(session_language),
        "campaign_context": (
            context_str
            + "\n\n--- CURRENT GAME STATE ---\n"
            + state_str
            + scene_loop_guard
            + ("\n\n" + verdict_guard if verdict_guard else "")
        ),
        "era_context": setting_override + " " + era_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": extract_last_turn_ban(db),
    }

    full_text = ""
    async for chunk in chain.astream(prompt_vars):
        full_text += chunk
        yield f"data: {_json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"

    logger.info("stream_chat_logic(): full streamed LLM response preview: %r", full_text[:4000])
    # ── Post-stream: parse full JSON, apply state, fire image gen ────────────
    result = extract_json(full_text)

    from pathlib import Path as _Path
    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    import uuid as _uuid
    gen_id = _uuid.uuid4().hex
    _image_results[gen_id] = "pending"
    result["generation_id"] = gen_id
    result["image_url"] = None

    async def _generate_image_bg(gid, narrative, setting, era):
        try:
            from img_gen.comfy_client import ComfyClient
            _comfy = ComfyClient(_COMFY_BASE_URL)
            visual_history, char_visuals = "", ""
            if req.session_id in active_dbs:
                _db = active_dbs[req.session_id]
                _cur = _db.conn.cursor()
                _cur.execute("SELECT value FROM kv_store WHERE key='visual_history'")
                r = _cur.fetchone(); visual_history = r["value"] if r else ""
                _cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'")
                rows = _cur.fetchall()
                if rows:
                    char_visuals = "ESTABLISHED CHARACTERS:\n" + "\n".join(f"- {r['value']}" for r in rows)
            from utils.helpers import build_scene_prompt
            scene_prompt = build_scene_prompt(narrative, era=era, setting=setting,
                                              visual_history=visual_history, char_visuals=char_visuals)
            with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as f:
                body = _json.load(f)
            body["params"]["prompt"] = scene_prompt
            img_result = _comfy.generate(body)
            _image_results[gid] = img_result["image_url"]
        except Exception as e:
            logger.warning(f"Image generation failed [{gid}]: {e}")
            _image_results[gid] = None

    asyncio.create_task(_generate_image_bg(gen_id, result.get("narrative", ""), setting_override, era_override))

    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})

    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row = cur.fetchone()
    turn_count = int(row["value"]) + 1 if row else 1
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)", (str(turn_count),))
    db.conn.commit()
    if turn_count % 5 == 0:
        asyncio.create_task(compress_story(db))

    apply_state_updates(db, result)
    yield f"data: {_json.dumps({'type': 'done', 'payload': result}, ensure_ascii=False)}\n\n"
