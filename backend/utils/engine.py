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
from utils.helpers import read_prompt, extract_json, get_chat_history, get_llm 

logger = logging.getLogger("keeper_ai.engine")

from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

# _COMFY_BASE_URL = "https://jacksonville-arms-latest-johnson.trycloudflare.com"  # update as needed
# _COMFY_BASE_URL = "https://provided-feeds-pipe-avatar.trycloudflare.com"  # update as needed

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

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
    llm = get_llm(temperature=0.7)
    chain = PromptTemplate.from_template(read_prompt("character_gen.txt")) | llm
    response_text = chain.invoke({
        "language": req.language,
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected"
    })
    return extract_json(response_text)

async def _compress_story(db: SessionDB):
    """Compress story history into a rolling digest stored in kv_store."""
    try:
        # Pull last 30 chat events for compression
        all_events = db.list_events(limit=60)
        chat_lines = []
        for e in all_events:
            if e.get("event_type") == "CHAT":
                p = e.get("payload", {})
                chat_lines.append(f"{p.get('role','?').upper()}: {p.get('content','')}")
        
        if len(chat_lines) < 4:
            return  # Not enough to compress yet
        
        full_history = "\n\n".join(chat_lines)
        
        cur = db.conn.cursor()
        cur.execute("SELECT value FROM kv_store WHERE key='story_digest'")
        prev = cur.fetchone()
        prev_digest = prev["value"] if prev else "(none yet)"
        
        cur.execute("SELECT value FROM kv_store WHERE key='language'")
        row = cur.fetchone()
        lang = row["value"] if row else "ru"
        
        compression_prompt = f"""You are a scribe summarizing a Call of Cthulhu session for continuity.
Language: {lang}. Write the digest in this language.

PREVIOUS DIGEST (events before this batch):
{prev_digest}

RECENT SESSION EXCHANGES:
{full_history[-6000:]}

Write a STORY DIGEST — a compact, factual record of what has happened in this session.
Format: numbered bullet points, past tense.
Cover: locations visited, NPCs encountered (what was learned from/about them), clues found, 
player decisions and their outcomes, plot threads advanced, any deaths/san loss/injuries.
DO NOT speculate. Only record what actually happened in the exchanges above.
Maximum 220 words. No preamble. Start directly with "1."
DIGEST:"""

        compress_llm = get_llm(temperature=0.2)
        digest = compress_llm.invoke(compression_prompt).strip()
        
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('story_digest', ?)", (digest,))
        db.conn.commit()
        logger.info(f"Story digest compressed: {len(digest)} chars")
    except Exception as e:
        logger.warning(f"Story compression failed: {e}")

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
        query_text = req.picked_seed          # full scenario text → sharp RAG
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

    # 3. PROACTIVE RAG: Pull Scenario Atoms, SYNTHESIZE a blueprint, then SAVE TO DB
    scenario_atoms_text = ""
    if scen_db:
        logger.info(f"Querying Scenario DB for starting atoms using: '{query_text}'")
        # Pull more atoms so synthesis has richer material to remix


        # Pull MORE candidates than needed, then randomly sample
        # This breaks the "same 6 atoms every time" loop
        scen_docs = scen_db.similarity_search(query_text, k=15)

        # Split: 2 closely relevant + 2 randomly picked from the rest + 1 from a DIFFERENT theme
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
        # Pick one that ISN'T already in our set
        existing_contents = {d.page_content for d in mixed_docs}
        cross_pick = next((d for d in cross_docs if d.page_content not in existing_contents), None)
        if cross_pick:
            mixed_docs.append(cross_pick)

        random.shuffle(mixed_docs)  # randomize order so model doesn't privilege first atoms

        atoms = [f"ATOM {i+1}:\n{doc.page_content.strip()}" for i, doc in enumerate(mixed_docs)]
        raw_atoms_text = "\n\n".join(atoms)

        # --- SYNTHESIS STEP: build a unique scenario blueprint ---
        logger.info("Synthesizing unique scenario blueprint from atoms...")
        lang = req.language if hasattr(req, 'language') else 'ru'
        try:
            synth_llm = get_llm(temperature=0.9)
            synth_chain = PromptTemplate.from_template(read_prompt("scenario_gen.txt")) | synth_llm
            synth_raw = synth_chain.invoke({
                "themes": themes_str,
                "era_context": era_context,
                "language": lang,
                "atoms": raw_atoms_text
            })
            blueprint = extract_json(synth_raw)
            # Serialize blueprint into a readable context block for the Keeper
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
        # ---------------------------------------------------------

        db.log_event("SCENARIO_GENERATED", {"query": query_text})

        # --- Populate DB tables from blueprint skeleton ---
        if isinstance(blueprint, dict):
            # Locations
            loc_id_map = {}  # name -> id, for clue linking
            for loc in blueprint.get("locations", []):
                lid = db.upsert_location(
                    name=loc.get("name", "Unknown"),
                    description=loc.get("description", ""),
                    tags=loc.get("tags", "")
                )
                loc_id_map[loc.get("name", "")] = lid

            # NPCs
            for npc in blueprint.get("npcs", []):
                role = npc.get("role", "neutral")
                kind = "ENEMY" if "enemy" in role else "NPC"
                db.upsert_actor(
                    kind=kind,
                    name=npc.get("name", "Unknown"),
                    description=npc.get("description", ""),
                    notes=npc.get("secret", "")
                )

            # Clues (hidden by default — Keeper reveals them)
            for clue in blueprint.get("clues", []):
                loc_name = clue.get("location", "")
                db.upsert_clue(
                    title=clue.get("title", "Clue"),
                    content=clue.get("content", ""),
                    status="hidden",
                    location_id=loc_id_map.get(loc_name)
                )

            # Plot threads
            for thread in blueprint.get("plot_threads", []):
                db.upsert_thread(
                    name=thread.get("name", "Thread"),
                    stakes=thread.get("stakes", ""),
                    max_progress=thread.get("steps", 4)
                )
        
        # scenario_setting = blueprint.get('era_and_setting', 'Lovecraftian Horror Arcane Lore for a Call of Cthulhu') if isinstance(blueprint, dict) else 'Lovecraftian Horror Arcane Lore for a Call of Cthulhu'
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

        # ── DEBUG: verify what's actually stored ──────────────────────────────────────
        print(f"==>> [SESSION START] era_context passed in : {repr(req.era_context)}")
        print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
        print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
        print(f"==>> [SESSION START] themes_str            : {repr(themes_str)}")
        # ─────────────────────────────────────────────────────────────────────────────

    # 4. Simple trigger message
    start_msg = "Начни историю. Опиши стартовую локацию." if req.language == 'ru' else "Start the story. Describe the starting location."
    return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))


async def generate_avatar_logic(req) -> dict:
    from pathlib import Path as _Path

    # _COMFY_BASE_URL = "https://provided-feeds-pipe-avatar.trycloudflare.com"
    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    # Build portrait prompt
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
    llm = get_llm(temperature=0.3)
    portrait_prompt = (tpl | llm).invoke({
        "era": req.era_context or "1920s Lovecraftian Horror",
        "occupation": req.occupation,
        "description": req.physical_description
    }).strip()
    portrait_prompt = " ".join(l.strip() for l in portrait_prompt.splitlines() if l.strip())

    logger.info(f"Avatar prompt for {req.name}: {portrait_prompt}")

    try:
        from img_gen.comfy_client import ComfyClient
        _comfy = ComfyClient(_COMFY_BASE_URL)
        with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as _f:
            _body = json.load(_f)
        _body["params"]["prompt"] = portrait_prompt
        _body["params"]["width"] = 480
        _body["params"]["height"] = 640
        img_result = _comfy.generate(_body)
        return {
            "image_url": img_result["image_url"],
            "portrait_prompt": portrait_prompt
        }
    except Exception as e:
        logger.warning(f"Avatar generation failed for {req.name}: {e}")
        return {"image_url": None, "portrait_prompt": portrait_prompt}
    
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
    is_start_msg = "Начни историю" in req.message or "Start the story" in req.message
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

    # 5. Invoke LLM
    llm = get_llm(temperature=req.temperature)
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt")) | llm
    
    # Send to LLM
        # Fetch setting override from blueprint if available
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row_setting = cur.fetchone()
    setting_override = row_setting["value"] if row_setting else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row_era = cur.fetchone()
    era_override = row_era["value"] if row_era else "1920s Lovecraftian Horror"
    
    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row_lang = cur.fetchone()
    session_language = row_lang["value"] if row_lang else "en"

    # ── DEBUG ──────────────────────────────────────────────────────────────────────
    print(f"==>> [CHAT] setting_override read from DB : {repr(setting_override)}")
    print(f"==>> [CHAT] era_override read from DB     : {repr(era_override)}")
    # ─────────────────────────────────────────────────────────────────────────────

    # Detect VERDICT messages and inject an explicit anti-repetition guard
    is_verdict = "[SYSTEM MESSAGE]" in req.message and "VERDICT" in req.message
    if is_verdict:
        verdict_guard = (
            "\n\n⚠ VERDICT RECEIVED. STRICT RULES FOR THIS RESPONSE:\n"
            "- Write 1–3 sentences MAXIMUM.\n"
            "- Describe ONLY what changed due to this roll result.\n"
            "- DO NOT reproduce or paraphrase any prior narrative.\n"
            "- DO NOT re-describe the location, NPCs, or atmosphere.\n"
            "- Start mid-action, from the moment the roll resolves.\n"
        )
    else:
        verdict_guard = ""

    # ── DYNAMIC BAN LIST: extract key phrases from last Keeper turn ────────────
    last_turn_ban = ""
    last_keeper_events = [
        e for e in db.list_events(limit=6)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ]
    if last_keeper_events:
        last_narrative = last_keeper_events[0].get("payload", {}).get("content", "")
        # Extract first 5 sentences as banned phrases
        import re as _re
        sentences = [s.strip() for s in _re.split(r'[.!?。]', last_narrative) if len(s.strip()) > 20]
        if sentences:
            banned = sentences[:5]
            last_turn_ban = (
                "PHRASES FROM YOUR PREVIOUS RESPONSE (DO NOT REUSE OR PARAPHRASE THESE):\n"
                + "\n".join(f'- "{s[:80]}"' for s in banned)
            )
    # ────────────────────────────────────────────────────────────────────────────

    response_text = chain.invoke({
        "language": session_language,
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str + verdict_guard,
        "era_context": setting_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": last_turn_ban,
    })
    
    result = extract_json(response_text)

    # ── IMAGE GENERATION ──────────────────────────────────────────────────────────
    import json as _json
    from pathlib import Path as _Path

    
    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    def _build_scene_prompt(narrative: str, era: str = "", setting: str = "", visual_history: str = "", char_visuals: str = "") -> str:
        tpl = PromptTemplate.from_template(
            "You are a prompt engineer for Flux, a natural-language image model. "
            "Your job is to generate a consistent visual description across a series of scenes.\n\n"
            "Setting/Era context: {era}, {setting}\n\n"
            "{char_visuals}\n\n"
            "PREVIOUS SCENE DESCRIPTIONS (for visual consistency):\n{visual_history}\n\n"
            "STEP 1 — Read the ENTIRE narrative. Identify the single element that carries the most narrative weight.\n"
            "STEP 2 — Frame it as a close or medium shot. That element must be the visual centerpiece.\n"
            "STEP 3 — Write 2-3 short natural sentences. If characters from the established list appear, "
            "describe them using their established appearance. Match era and setting in every detail.\n"
            "STEP 4 — Append exactly: 'Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.'\n\n"
            "RULES:\n"
            "- English only. No character names. No smell or sound.\n"
            "- Output ONLY the final prompt, nothing else\n\n"
            "NARRATIVE:\n{narrative}\n\nPROMPT:"
        )
        llm = get_llm(temperature=0.3)
        raw = (tpl | llm).invoke({
            "narrative": narrative, "era": era, "setting": setting,
            "visual_history": visual_history or "No previous scenes yet.",
            "char_visuals": char_visuals or ""
        }).strip()
        return " ".join(l.strip() for l in raw.splitlines() if l.strip())

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

            scene_prompt = _build_scene_prompt(
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
        asyncio.create_task(_compress_story(db))
        logger.info(f"Story compression triggered at turn {turn_count}")
    
    # 6. Apply State Updates
    if state_updates := result.get("state_updates"):

        # --- PC stat changes ---
        if target_name := state_updates.get("character_name"):
            all_actors = db.list_actors("PC") + db.list_actors("NPC") + db.list_actors("ENEMY")
            target = next((a for a in all_actors if target_name.lower() in a["name"].lower()), None)
            if target:
                changes = {}
                hp_change  = int(state_updates.get("hp_change", 0) or 0)
                san_change = int(state_updates.get("sanity_change", 0) or 0)
                mp_change  = int(state_updates.get("mp_change", 0) or 0)

                if hp_change != 0 or san_change != 0 or mp_change != 0:
                    new_hp  = max(0, (target.get("hp")  or 0) + hp_change)
                    new_san = max(0, (target.get("san") or 0) + san_change)
                    new_mp  = max(0, (target.get("mp")  or 0) + mp_change)

                    # Derive status
                    status = target.get("status", "ok")
                    if new_hp == 0:
                        status = "dead"
                    elif new_san == 0:
                        status = "insane"
                    elif hp_change < 0:
                        status = "injured"

                    db.upsert_actor(
                        actor_id=target["id"], kind=target["kind"], name=target["name"],
                        hp=new_hp, san=new_san, mp=new_mp, status=status
                    )
                    changes = {
                        "actor": target["name"],
                        **({f"hp": f"{hp_change:+} → {new_hp}"} if hp_change else {}),
                        **({f"san": f"{san_change:+} → {new_san}"} if san_change else {}),
                        **({f"mp": f"{mp_change:+} → {new_mp}"} if mp_change else {}),
                        "status": status
                    }
                    db.log_event("STATE_UPDATE", changes)
                    # Return absolute values so frontend can SET instead of delta-apply
                    result["updated_actor"] = {
                        "name": target["name"],
                        "hp": new_hp,
                        "san": new_san,
                        "mp": new_mp,
                        "status": status,
                    }

                # Inventory
                if item_add := state_updates.get("inventory_add", ""):
                    db.upsert_clue(title=item_add, content=f"Carried by {target['name']}", status="found")
                    db.log_event("INVENTORY", {"actor": target["name"], "added": item_add})

                if item_remove := state_updates.get("inventory_remove", ""):
                    db.log_event("INVENTORY", {"actor": target["name"], "removed": item_remove})

        # --- Location change ---
        if new_location := state_updates.get("location_name", ""):
            existing = db.get_location_by_name(new_location)
            if existing:
                loc_id = existing["id"]
            else:
                loc_id = db.upsert_location(name=new_location, description=state_updates.get("location_description", ""))
            # Move all PCs to new location
            for pc in db.list_actors("PC"):
                db.upsert_actor(actor_id=pc["id"], kind="PC", name=pc["name"], location_id=loc_id)
            db.log_event("LOCATION_CHANGE", {"location": new_location})

        # --- Clue discovered ---
        if clue_found := state_updates.get("clue_found", ""):
            clues = db.list_clues(status="hidden")
            match = next((c for c in clues if clue_found.lower() in c["title"].lower()), None)
            if match:
                db.upsert_clue(
                    clue_id=match["id"], title=match["title"],
                    content=match["content"], status="found"
                )
            else:
                # Brand new clue not in blueprint — add it
                db.upsert_clue(title=clue_found, content=state_updates.get("clue_content", ""), status="found")
            db.log_event("CLUE_FOUND", {"clue": clue_found})

        # --- Thread progress ---
        if thread_name := state_updates.get("thread_progress", ""):
            threads = db.list_threads()
            match = next((t for t in threads if thread_name.lower() in t["name"].lower()), None)
            if match:
                db.upsert_thread(
                    thread_id=match["id"], name=match["name"],
                    progress=min(match["progress"] + 1, match["max_progress"]),
                    max_progress=match["max_progress"], stakes=match.get("stakes", "")
                )
                db.log_event("THREAD_PROGRESS", {"thread": match["name"], "progress": match["progress"] + 1})

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
    row = cur.fetchone(); session_language = row["value"] if row else "ru"

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

    is_start_msg = "Начни историю" in req.message or "Start the story" in req.message
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

    is_verdict = "[SYSTEM MESSAGE]" in req.message and "VERDICT" in req.message
    verdict_guard = (
        "\n\n⚠ VERDICT RECEIVED. STRICT RULES FOR THIS RESPONSE:\n"
        "- Write 1–3 sentences MAXIMUM.\n"
        "- Describe ONLY what changed due to this roll result.\n"
        "- DO NOT reproduce or paraphrase any prior narrative.\n"
        "- Start mid-action, from the moment the roll resolves.\n"
    ) if is_verdict else ""

    last_keeper_events = [
        e for e in db.list_events(limit=6)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ]
    last_turn_ban = ""
    if last_keeper_events:
        import re as _re
        last_narrative = last_keeper_events[0].get("payload", {}).get("content", "")
        sentences = [s.strip() for s in _re.split(r'[.!?。]', last_narrative) if len(s.strip()) > 20]
        if sentences:
            last_turn_ban = (
                "PHRASES FROM YOUR PREVIOUS RESPONSE (DO NOT REUSE OR PARAPHRASE THESE):\n"
                + "\n".join(f'- "{s[:80]}"' for s in sentences[:5])
            )

    # ── Stream tokens from LLM ───────────────────────────────────────────────
    llm = get_llm(temperature=req.temperature)
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt")) | llm

    prompt_vars = {
        "language": session_language,
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str + verdict_guard,
        "era_context": setting_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": last_turn_ban,
    }

    full_text = ""
    async for chunk in chain.astream(prompt_vars):
        full_text += chunk
        yield f"data: {_json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"

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
        asyncio.create_task(_compress_story(db))

    if state_updates := result.get("state_updates"):
        if target_name := state_updates.get("character_name"):
            all_actors = db.list_actors("PC") + db.list_actors("NPC") + db.list_actors("ENEMY")
            target = next((a for a in all_actors if target_name.lower() in a["name"].lower()), None)
            if target:
                hp_change  = int(state_updates.get("hp_change", 0) or 0)
                san_change = int(state_updates.get("sanity_change", 0) or 0)
                mp_change  = int(state_updates.get("mp_change", 0) or 0)
                if hp_change or san_change or mp_change:
                    new_hp  = max(0, (target.get("hp") or 0) + hp_change)
                    new_san = max(0, (target.get("san") or 0) + san_change)
                    new_mp  = max(0, (target.get("mp") or 0) + mp_change)
                    status = "dead" if new_hp == 0 else "insane" if new_san == 0 else "injured" if hp_change < 0 else target.get("status", "ok")
                    db.upsert_actor(actor_id=target["id"], kind=target["kind"], name=target["name"],
                                    hp=new_hp, san=new_san, mp=new_mp, status=status)
                    db.log_event("STATE_UPDATE", {"actor": target["name"], "status": status})
                    # Return absolute values so frontend can SET instead of delta-apply
                    result["updated_actor"] = {
                        "name": target["name"],
                        "hp": new_hp,
                        "san": new_san,
                        "mp": new_mp,
                        "status": status,
                    }
                if item_add := state_updates.get("inventory_add", ""):
                    db.upsert_clue(title=item_add, content=f"Carried by {target['name']}", status="found")
                if item_remove := state_updates.get("inventory_remove", ""):
                    db.log_event("INVENTORY", {"actor": target["name"], "removed": item_remove})

        if new_location := state_updates.get("location_name", ""):
            existing = db.get_location_by_name(new_location)
            loc_id = existing["id"] if existing else db.upsert_location(
                name=new_location, description=state_updates.get("location_description", ""))
            for pc in db.list_actors("PC"):
                db.upsert_actor(actor_id=pc["id"], kind="PC", name=pc["name"], location_id=loc_id)
            db.log_event("LOCATION_CHANGE", {"location": new_location})

        if clue_found := state_updates.get("clue_found", ""):
            clues = db.list_clues(status="hidden")
            match = next((c for c in clues if clue_found.lower() in c["title"].lower()), None)
            if match:
                db.upsert_clue(clue_id=match["id"], title=match["title"], content=match["content"], status="found")
            else:
                db.upsert_clue(title=clue_found, content=state_updates.get("clue_content", ""), status="found")
            db.log_event("CLUE_FOUND", {"clue": clue_found})

        if thread_name := state_updates.get("thread_progress", ""):
            threads = db.list_threads()
            match = next((t for t in threads if thread_name.lower() in t["name"].lower()), None)
            if match:
                db.upsert_thread(thread_id=match["id"], name=match["name"],
                                 progress=min(match["progress"] + 1, match["max_progress"]),
                                 max_progress=match["max_progress"], stakes=match.get("stakes", ""))
                db.log_event("THREAD_PROGRESS", {"thread": match["name"]})

    yield f"data: {_json.dumps({'type': 'done', 'payload': result}, ensure_ascii=False)}\n\n"
