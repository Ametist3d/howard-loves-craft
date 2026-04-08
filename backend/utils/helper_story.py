import json
import logging
from pathlib import Path

from utils.db_session import SessionDB
from utils.helpers import get_language_name, get_llm, kv_get
from langchain_core.prompts import PromptTemplate
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

logger = logging.getLogger("keeper_ai.helpers.story")


def infer_scene_stall_level(db: SessionDB) -> int:
    """
    Rough anti-loop signal: count recent user turns since the latest meaningful progress event.
    0 = fresh, 1 = mildly stalled, 2 = clearly stalled.
    """
    events = db.list_events(limit=18)
    turns_since_progress = 0

    # list_events() returns chronological order (oldest -> newest),
    # so we must scan backwards from the latest events.
    for e in reversed(events):
        et = e.get("event_type")
        if et in ("LOCATION_CHANGE", "CLUE_FOUND", "THREAD_PROGRESS"):
            break
        if et == "CHAT" and e.get("payload", {}).get("role") == "User":
            turns_since_progress += 1

    if turns_since_progress >= 4:
        return 2
    if turns_since_progress >= 2:
        return 1
    return 0


def build_stall_forcing_guard(db: SessionDB) -> str:
    stall = infer_scene_stall_level(db)
    if stall <= 0:
        return ""

    if stall == 1:
        return (
            "\n\nSTALL ESCALATION NOTICE\n"
            "The scene is starting to stall.\n"
            "This response MUST include at least one concrete payoff:\n"
            "- a clue fully revealed\n"
            "- a thread advanced\n"
            "- a location change\n"
            "- a named NPC reaction\n"
            "- a visible threat or irreversible choice\n"
            "Do not answer with another partial step toward the same discovery."
        )

    return (
        "\n\nHARD STALL OVERRIDE\n"
        "The scene is stalled.\n"
        "This response MUST NOT be another setup step, atmosphere beat, or 'you realize there may be a way'.\n"
        "You MUST do one of the following NOW:\n"
        "- reveal the core fact the investigators are pursuing\n"
        "- trigger immediate threat contact\n"
        "- force an irreversible decision\n"
        "- complete the current discovery and update the thread\n"
        "Do not suggest 'check again', 'observe further', 'try to understand more', or equivalent soft continuation."
    )


def build_scene_loop_guard(db: SessionDB) -> str:
    events = db.list_events(limit=14)
    user_actions = []
    keeper_bits = []

    for e in events:
        if e.get("event_type") != "CHAT":
            continue
        payload = e.get("payload", {})
        role = payload.get("role", "")
        content = (payload.get("content", "") or "").strip()

        if role == "User" and content:
            user_actions.append(content[:120])
        elif role == "Keeper" and content:
            keeper_bits.append(content[:220])

    recent_user = user_actions[-4:]
    recent_keeper = keeper_bits[-4:]
    keeper_blob = "\n".join(recent_keeper).lower()

    corridor_markers = [
        "door", "corridor", "hallway", "stairs", "second floor", "room",
        "двер", "корид", "сход", "кімнат", "поверх"
    ]
    corridor_hits = sum(1 for m in corridor_markers if m in keeper_blob)

    extra = (
        "\nIf the recent pattern is repeated questioning, social probing, or Psychology against the same NPC, "
        "do NOT answer with another vague read like 'they seem nervous' or 'they hide something'. "
        "Instead, the next response MUST create concrete movement through at least one of:\n"
        "- a direct contradiction or slip\n"
        "- a named clue, place, or person\n"
        "- an interruption, witness, or outside event\n"
        "- the NPC ending the exchange, leaving, threatening, or making a mistake\n"
        "- a new actionable lead that changes what the investigators do next\n"
        "A successful social/psychology payoff must yield actionable information, leverage, or a visible change in the scene."
    )

    forced = ""
    if corridor_hits >= 3:
        forced = (
            "\nHARD LOOP DETECTED: repeated room/corridor/door progression."
            "\nThe next response MUST NOT introduce another door, corridor, staircase, or adjacent room as the main development."
            "\nInstead, the next response MUST do one of the following immediately:"
            "\n- reveal the missing friend's fate or exact last known action"
            "\n- put a speaking NPC, witness, or hostile cultist on-screen"
            "\n- deliver a clue that directly points to the park and explains why"
            "\n- trigger immediate danger with a concrete consequence"
            "\n- force a decision with a cost"
        )

    return (
        "RECENT SCENE LOOP WARNING\n"
        "Do not repeat the same pattern again.\n"
        "Recent user actions:\n"
        + "\n".join(f"- {x}" for x in recent_user)
        + "\nRecent keeper outcomes:\n"
        + "\n".join(f"- {x}" for x in recent_keeper)
        + "\nThe next response MUST materially change the scene."
        + "\nForbidden next-step patterns include:"
        + "\n- another vague reaction to the same object"
        + "\n- another partial understanding of the same clue"
        + "\n- another atmospheric warning without consequence"
        + "\n- another 'you may now try X' after X was effectively already tried"
        + forced
        + "\nYou must either complete the current discovery, reveal a concrete new fact, trigger danger, or force a choice."
        + extra
    )


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
        "STEP 1 — Read the ENTIRE narrative. Identify key element or location that carries the most narrative weight.\n"
        "STEP 2 - Start with `Painterly digital illustration of...` \n"
        "STEP 3 — if narrating location - write description to recreate that environment, if narrating subject - frame it as a close or medium shot. That element must be the visual centerpiece.\n"
        "STEP 4 — Write 2-3 short natural sentences. If characters from the established list appear, "
        "describe them using their established appearance. Match era and setting in every detail.\n"
        "STEP 5 — Append exactly: 'Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.'\n\n"
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
        lang = kv_get(cur, "language", "en")
        language_name = get_language_name(lang)

        compression_prompt = (
            f"You are a scribe summarizing a Call of Cthulhu session for continuity.\n"
            f"Language: {language_name}. Write the digest in this language.\n\n"
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
        " - Start with `Painterly digital illustration of...` \n"
        "- 1-2 sentences describing: most prominent character features matching character back-story, clothing appropriate to the era, and setting, posture/mood\n"
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
