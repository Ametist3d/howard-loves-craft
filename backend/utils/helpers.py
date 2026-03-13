import os
import re
import json
import logging
from typing import Optional
from utils.db_session import SessionDB
from langchain_core.prompts import PromptTemplate
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

logger = logging.getLogger("keeper_ai.helpers")


# ─────────────────────────────────────────────
# File / JSON helpers
# ─────────────────────────────────────────────

def read_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


def extract_json(text: str) -> dict:
    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)

    # Isolate the JSON block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]

    # Fix LLM outputting "-1d6" instead of integers in JSON values
    text = re.sub(r':\s*([-+]?\d+)[dD]\d+', r': \1', text)

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error intercepted: {e}. Attempting Regex Fallback.")
        narrative_match = re.search(
            r'"narrative"\s*:\s*"(.*?)"(?:\s*,|\s*\n\s*"suggested_actions")',
            text, re.DOTALL | re.IGNORECASE
        )
        narrative = narrative_match.group(1) if narrative_match else text
        narrative = narrative.replace('\n', ' ').replace('\\n', '\n')
        return {
            "narrative": narrative + "\n\n*(OOC: The Keeper's connection wavered, some state updates may have been lost.)*",
            "suggested_actions": ["Continue"],
            "state_updates": None
        }


# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────

def get_chat_history(db: SessionDB, limit: int = 10) -> str:
    """
    Returns recent chat history for LLM context.
    Only the last 4 messages (2 full turns) are included — the story digest
    covers everything older, so we don't need more raw history here.
    Keeper messages are summarised to their last sentence to prevent
    the model from parroting prior prose verbatim.
    """
    events = db.list_events(limit=limit * 2)
    pairs = []
    for e in reversed(events):
        if e.get("event_type") == "CHAT":
            payload = e.get("payload", {})
            role = payload.get("role", "Unknown")
            content = payload.get("content", "")
            pairs.append((role, content))

    # Keep only the last 4 messages (2 user + 2 keeper turns max)
    recent_pairs = pairs[:4]

    chat_lines = []
    for role, content in recent_pairs:
        if role.lower() == "keeper":
            # Summarise to last sentence — full text is a copy-template Gemma will parrot
            if len(content) > 150:
                last_sentence = content.rstrip().rsplit(".", 1)[-1].strip()
                if not last_sentence or len(last_sentence) < 10:
                    last_sentence = content[-120:].strip()
                content = f"[...сцена описана ранее...] {last_sentence}"
            chat_lines.append(f"KEEPER: {content}")
        else:
            # Player messages are short actions — keep them full
            chat_lines.append(f"USER: {content}")

    return "\n\n".join(chat_lines)


# ─────────────────────────────────────────────
# kv_store convenience
# ─────────────────────────────────────────────

def kv_get(cur, key: str, default: str = "") -> str:
    """Read a single value from kv_store by key."""
    cur.execute("SELECT value FROM kv_store WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


# ─────────────────────────────────────────────
# Verdict guard (injected into campaign_context on dice results)
# ─────────────────────────────────────────────

def build_verdict_guard(message: str) -> str:
    """
    Returns an anti-repetition instruction block when a VERDICT system message
    is detected, empty string otherwise.
    """
    if "[SYSTEM MESSAGE]" in message and "VERDICT" in message:
        return (
            "\n\n⚠ VERDICT RECEIVED. STRICT RULES FOR THIS RESPONSE:\n"
            "- Write 1–3 sentences MAXIMUM.\n"
            "- Describe ONLY what changed due to this roll result.\n"
            "- DO NOT reproduce or paraphrase any prior narrative.\n"
            "- DO NOT re-describe the location, NPCs, or atmosphere.\n"
            "- Start mid-action, from the moment the roll resolves.\n"
        )
    return ""


# ─────────────────────────────────────────────
# Dynamic ban list (phrases from last Keeper turn)
# ─────────────────────────────────────────────

def extract_last_turn_ban(db: SessionDB) -> str:
    """
    Extracts the first 5 sentences of the most recent Keeper message and
    returns a formatted ban block instructing the model not to reuse them.
    Returns empty string if no previous Keeper turn exists.
    """
    last_keeper_events = [
        e for e in db.list_events(limit=6)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ]
    if not last_keeper_events:
        return ""

    last_narrative = last_keeper_events[0].get("payload", {}).get("content", "")
    sentences = [s.strip() for s in re.split(r'[.!?。]', last_narrative) if len(s.strip()) > 20]
    if not sentences:
        return ""

    banned = sentences[:5]
    return (
        "PHRASES FROM YOUR PREVIOUS RESPONSE (DO NOT REUSE OR PARAPHRASE THESE):\n"
        + "\n".join(f'- "{s[:80]}"' for s in banned)
    )


# ─────────────────────────────────────────────
# State string builder
# ─────────────────────────────────────────────

def build_state_str(pack: dict) -> str:
    """Formats the prompt-state pack dict into a clean string block for the LLM."""
    return (
        f"INVESTIGATORS:\n{pack.get('investigators_text')}\n\n"
        f"KNOWN NPCs:\n{pack.get('npcs_text')}\n\n"
        f"CURRENT LOCATION:\n{pack.get('location_text')}\n\n"
        f"PLOT THREADS:\n{pack.get('threads_text')}\n\n"
        f"DISCOVERED CLUES:\n{pack.get('clues_text')}"
    )


# ─────────────────────────────────────────────
# Scene image prompt builder
# ─────────────────────────────────────────────

def build_scene_prompt(
    narrative: str,
    era: str = "",
    setting: str = "",
    visual_history: str = "",
    char_visuals: str = "",
) -> str:
    """
    Generates a Flux image prompt from the current scene narrative.
    Uses a low-temperature Gemma call for consistency.
    Imported here to keep engine.py free of nested function definitions.
    """
    from langchain_core.prompts import PromptTemplate

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
        "narrative": narrative,
        "era": era,
        "setting": setting,
        "visual_history": visual_history or "No previous scenes yet.",
        "char_visuals": char_visuals or "",
    }).strip()
    return " ".join(line.strip() for line in raw.splitlines() if line.strip())


# ─────────────────────────────────────────────
# Story digest (rolling LLM compression)
# ─────────────────────────────────────────────

async def compress_story(db: SessionDB) -> None:
    """
    Compresses the full chat history into a rolling story digest stored in
    kv_store['story_digest']. Called every N turns from handle_chat_logic.
    The digest is injected at the top of every prompt as the model's primary
    long-term memory anchor.
    """
    try:
        all_events = db.list_events(limit=60)
        chat_lines = []
        for e in all_events:
            if e.get("event_type") == "CHAT":
                p = e.get("payload", {})
                chat_lines.append(f"{p.get('role', '?').upper()}: {p.get('content', '')}")

        if len(chat_lines) < 4:
            return  # Not enough material to compress yet

        full_history = "\n\n".join(chat_lines)

        cur = db.conn.cursor()
        prev_digest = kv_get(cur, "story_digest", "(none yet)")
        lang = kv_get(cur, "language", "ru")

        compression_prompt = (
            f"You are a scribe summarizing a Call of Cthulhu session for continuity.\n"
            f"Language: {lang}. Write the digest in this language.\n\n"
            f"PREVIOUS DIGEST (events before this batch):\n{prev_digest}\n\n"
            f"RECENT SESSION EXCHANGES:\n{full_history[-6000:]}\n\n"
            f"Write a STORY DIGEST — a compact, factual record of what has happened in this session.\n"
            f"Format: numbered bullet points, past tense.\n"
            f"Cover: locations visited, NPCs encountered (what was learned from/about them), clues found,\n"
            f"player decisions and their outcomes, plot threads advanced, any deaths/san loss/injuries.\n"
            f"DO NOT speculate. Only record what actually happened in the exchanges above.\n"
            f"Maximum 220 words. No preamble. Start directly with '1.'\n"
            f"DIGEST:"
        )

        llm = get_llm(temperature=0.2)
        digest = llm.invoke(compression_prompt).strip()

        cur.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('story_digest', ?)",
            (digest,)
        )
        db.conn.commit()
        logger.info(f"Story digest compressed: {len(digest)} chars")

    except Exception as e:
        logger.warning(f"Story compression failed: {e}")


# ─────────────────────────────────────────────
# LLM factory — provider-agnostic
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.7, *, streaming: bool = False):
    """
    Returns a LangChain LLM configured from .env.

    .env keys:
        LLM_PROVIDER   = "ollama" | "openai"          (default: ollama)
        OLLAMA_MODEL   = "gemma3:27b"                  (default)
        OLLAMA_BASE_URL= "http://localhost:11434"       (default)
        OPENAI_MODEL   = "gpt-4o"                      (default)
        OPENAI_API_KEY = "sk-..."                      (required for openai)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            streaming=streaming,
        ) | StrOutputParser()

    # Default: Ollama
    from langchain_ollama import OllamaLLM
    model = os.getenv("OLLAMA_MODEL", "gemma3:27b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

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

        # ─────────────────────────────────────────────
# State update applicator
# ─────────────────────────────────────────────

def apply_state_updates(db: SessionDB, result: dict) -> None:
    """
    Reads result['state_updates'] and applies all mechanical changes to the DB.
    Mutates result in-place to add 'updated_actor' if stats changed.
    """
    state_updates = result.get("state_updates")
    if not state_updates:
        return

    # --- PC stat changes ---
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
                if new_hp == 0:   status = "dead"
                elif new_san == 0: status = "insane"
                elif hp_change < 0: status = "injured"
                db.upsert_actor(actor_id=target["id"], kind=target["kind"], name=target["name"],
                                hp=new_hp, san=new_san, mp=new_mp, status=status)
                db.log_event("STATE_UPDATE", {"actor": target["name"], "status": status,
                    **({f"hp": f"{hp_change:+}→{new_hp}"} if hp_change else {}),
                    **({f"san": f"{san_change:+}→{new_san}"} if san_change else {}),
                    **({f"mp": f"{mp_change:+}→{new_mp}"} if mp_change else {}),
                })
                result["updated_actor"] = {
                    "name": target["name"], "hp": new_hp,
                    "san": new_san, "mp": new_mp, "status": status,
                }

            if item_add := state_updates.get("inventory_add", ""):
                db.upsert_clue(title=item_add, content=f"Carried by {target['name']}", status="found")
                db.log_event("INVENTORY", {"actor": target["name"], "added": item_add})
            if item_remove := state_updates.get("inventory_remove", ""):
                db.log_event("INVENTORY", {"actor": target["name"], "removed": item_remove})

    # --- Location change ---
    if new_location := state_updates.get("location_name", ""):
        existing = db.get_location_by_name(new_location)
        loc_id = existing["id"] if existing else db.upsert_location(
            name=new_location, description=state_updates.get("location_description", ""))
        for pc in db.list_actors("PC"):
            db.upsert_actor(actor_id=pc["id"], kind="PC", name=pc["name"], location_id=loc_id)
        db.log_event("LOCATION_CHANGE", {"location": new_location})

    # --- Clue discovered ---
    if clue_found := state_updates.get("clue_found", ""):
        clues = db.list_clues(status="hidden")
        match = next((c for c in clues if clue_found.lower() in c["title"].lower()), None)
        if match:
            db.upsert_clue(clue_id=match["id"], title=match["title"],
                           content=match["content"], status="found")
        else:
            db.upsert_clue(title=clue_found, content=state_updates.get("clue_content", ""), status="found")
        db.log_event("CLUE_FOUND", {"clue": clue_found})

    # --- Thread progress ---
    if thread_name := state_updates.get("thread_progress", ""):
        threads = db.list_threads()
        match = next((t for t in threads if thread_name.lower() in t["name"].lower()), None)
        if match:
            db.upsert_thread(thread_id=match["id"], name=match["name"],
                             progress=min(match["progress"] + 1, match["max_progress"]),
                             max_progress=match["max_progress"], stakes=match.get("stakes", ""))
            db.log_event("THREAD_PROGRESS", {"thread": match["name"], "progress": match["progress"] + 1})
            