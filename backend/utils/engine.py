import os
import logging
import json
from typing import Dict

# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest, ChatRequest
from utils.helpers import read_prompt, extract_json, get_chat_history 

logger = logging.getLogger("keeper_ai.engine")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)

# In-Memory Session Storage
active_dbs: Dict[str, SessionDB] = {}

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
    llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434", temperature=0.7)
    chain = PromptTemplate.from_template(read_prompt("character_gen.txt")) | llm
    response_text = chain.invoke({
        "language": req.language,
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
        scen_docs = scen_db.similarity_search(query_text, k=6)
        atoms = [f"ATOM {i+1}:\n{doc.page_content.strip()}" for i, doc in enumerate(scen_docs)]
        raw_atoms_text = "\n\n".join(atoms)

        # --- SYNTHESIS STEP: build a unique scenario blueprint ---
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
        
        scenario_setting = blueprint.get('era_and_setting', 'Lovecraftian Horror Arcane Lore for a Call of Cthulhu') if isinstance(blueprint, dict) else 'Lovecraftian Horror Arcane Lore for a Call of Cthulhu'
        
        cur = db.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (scenario_atoms_text,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (themes_str,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (era_context,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (req.picked_seed[:100],))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (req.language,))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)", 
            (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",))
        db.conn.commit()

    # 4. Simple trigger message
    start_msg = "Начни историю. Опиши стартовую локацию." if req.language == 'ru' else "Start the story. Describe the starting location."
    return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))


async def handle_chat_logic(req: ChatRequest) -> dict:
    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    db.log_event("CHAT", {"role": "User", "content": req.message})
    
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
    state_str = (
        f"INVESTIGATORS:\n{pack.get('investigators_text')}\n\n"
        f"KNOWN NPCs:\n{pack.get('npcs_text')}\n\n"
        f"CURRENT LOCATION:\n{pack.get('location_text')}\n\n"
        f"PLOT THREADS:\n{pack.get('threads_text')}\n\n"
        f"DISCOVERED CLUES:\n{pack.get('clues_text')}"
    )

    # 5. Invoke LLM
    llm = OllamaLLM(
        model="gemma3:27b",
        base_url="http://localhost:11434",
        temperature=req.temperature,
        num_ctx=16384 
    )
    chain = PromptTemplate.from_template(read_prompt("keeper_chat.txt")) | llm
    
    # Send to LLM
        # Fetch setting override from blueprint if available
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row_setting = cur.fetchone()
    setting_override = row_setting["value"] if row_setting else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row_lang = cur.fetchone()
    session_language = row_lang["value"] if row_lang else "en"

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

    response_text = chain.invoke({
        "language": session_language,
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str + verdict_guard,
        "era_context": setting_override,
        "history": get_chat_history(db),
        "action": req.message
    })
    
    result = extract_json(response_text)
    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})
    
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