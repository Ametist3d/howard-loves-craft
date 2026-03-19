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
from utils.helpers import read_prompt, extract_json, get_chat_history, get_llm, compress_story, build_scene_prompt, generate_avatar_logic, apply_state_updates
from utils.prompt_translate import ensure_translated_prompts, LANGUAGE_LABELS

logger = logging.getLogger("keeper_ai.engine")

from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL


# _COMFY_BASE_URL = "https://jacksonville-arms-latest-johnson.trycloudflare.com"  # update as needed
# _COMFY_BASE_URL = "https://provided-feeds-pipe-avatar.trycloudflare.com"  # update as needed

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/utils
_BACKEND_DIR = os.path.dirname(BASE_DIR)                # backend
DATA_DIR = os.path.join(_BACKEND_DIR, "data")           # backend/data
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")       # backend/data/sessions

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
    llm = get_llm(temperature=0.7)
    prompt_dir = ensure_translated_prompts("__pre_session__", req.language, filenames=("character_gen.txt",))
    chain = PromptTemplate.from_template(read_prompt("character_gen.txt", session_prompt_dir=prompt_dir)) | llm
    response_text = chain.invoke({
        "language": req.language,
        "language_name": LANGUAGE_LABELS.get(req.language, req.language),
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected"
    })
    return extract_json(response_text)

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

    prompt_dir = ensure_translated_prompts(session_id, req.language)
    logger.info(f"Prepared translated prompt set for session '{session_id}' in {req.language}: {prompt_dir}")

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
    lang = req.language if hasattr(req, 'language') else 'en'

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
            synth_chain = PromptTemplate.from_template(read_prompt("scenario_gen.txt", session_prompt_dir=prompt_dir)) | synth_llm
            synth_raw = synth_chain.invoke({
                "themes": themes_str,
                "era_context": era_context,
                "language": lang,
                "language_name": LANGUAGE_LABELS.get(lang, lang),
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
            blueprint = {}
            scenario_atoms_text = raw_atoms_text

    # ── SHARED: save everything to DB — runs for BOTH paths ───────────────────
    _dbg("FINAL scenario_atoms_text", scenario_atoms_text)
    _dbg("FINAL blueprint (first 1000)", json.dumps(blueprint, ensure_ascii=False)[:1000])
    logger.info(f"[SESSION] Final scenario_atoms_text (first 300): {scenario_atoms_text[:300]}")
    logger.info(f"[SESSION] language being stored: {repr(req.language)}")

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
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (str(req.language),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_dir', ?)", (str(prompt_dir),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_language', ?)", (str(req.language),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)",
        (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",))
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", str(era_context)))
    db.conn.commit()

    print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
    print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
    print(f"==>> [SESSION START] language stored        : {repr(req.language)}")

    # 4. Trigger opening narration
    start_msg = "Start the story. Describe the starting location."
    return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))
    
async def handle_chat_logic(req: ChatRequest) -> dict:
    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    db.log_event("CHAT", {"role": "User", "content": req.message})
    pcs = db.list_actors("PC")
    active_pcs = [a for a in pcs if a.get("status") not in ("dead", "insane")]
    inactive_pcs = [a for a in pcs if a.get("status") in ("dead", "insane")]

    if not active_pcs:
        return {
            "narrative": "💀 The story is over. No investigator remains able to continue.",
            "suggested_actions": [],
            "state_updates": None,
            "game_over": True,
        }

    if inactive_pcs:
        dead_names = {a["name"].lower() for a in inactive_pcs}
        msg_lower = req.message.lower()

        if any(name in msg_lower for name in dead_names):
            names = ", ".join(a["name"] for a in inactive_pcs if a["name"].lower() in msg_lower)
            return {
                "narrative": f"💀 {names} can no longer act. Their story is over.",
                "suggested_actions": [
                    f"Continue as {a['name']}" for a in active_pcs[:3]
                ],
                "state_updates": None,
                "game_over": False,
            }
    
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

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row_prompt_dir = cur.fetchone()
    session_prompt_dir = row_prompt_dir["value"] if row_prompt_dir and row_prompt_dir["value"] else None

    llm = get_llm(temperature=req.temperature)
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt", session_prompt_dir=session_prompt_dir)) | llm

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
    language_name = LANGUAGE_LABELS.get(session_language, session_language)

    print(f"==>> [CHAT] session_language={repr(session_language)}")
    
    response_text = chain.invoke({
        "language": session_language,
        "language_name": language_name,
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
    row = cur.fetchone(); session_language = row["value"] if row else "en"

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row = cur.fetchone(); session_prompt_dir = row["value"] if row and row["value"] else None

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
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt", session_prompt_dir=session_prompt_dir)) | llm

    prompt_vars = {
        "language": session_language,
        "language_name": LANGUAGE_LABELS.get(session_language, session_language),
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
        asyncio.create_task(compress_story(db))

    apply_state_updates(db, result)
    yield f"data: {_json.dumps({'type': 'done', 'payload': result}, ensure_ascii=False)}\n\n"
