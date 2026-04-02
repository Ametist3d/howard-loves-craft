import os
import re
import json
import logging
from typing import Optional, Any
from pathlib import Path
from utils.db_session import SessionDB
from langchain_core.prompts import PromptTemplate
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_CANDIDATE_DIRS = [
    CURRENT_DIR / "prompts",
    CURRENT_DIR.parent / "prompts",
    Path("/mnt/data/prompts"),
    CURRENT_DIR,
    Path("/mnt/data"),
]

logger = logging.getLogger("keeper_ai.helpers")


# ─────────────────────────────────────────────
# File / JSON helpers
# ─────────────────────────────────────────────
LANGUAGE_NAMES = {
    "ua": "Ukrainian",
    "en": "English",
    "ru": "Russian",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "hr": "Croatian",
}

def normalize_language_code(lang: str | None) -> str:
    raw = (lang or "en").strip().lower()
    return raw if raw else "en"

def get_language_name(lang: str | None) -> str:
    code = normalize_language_code(lang)
    return LANGUAGE_NAMES.get(code, code)

def build_translation_instruction(target_lang: str) -> str:
    language_name = get_language_name(target_lang)
    return (
        f"Translate into {language_name}. "
        "Preserve placeholders, JSON keys, and variable markers exactly. "
        "Do NOT translate proper nouns unless natural usage requires inflection only. "
        "Output only the translated text."
    )

def read_prompt(filename: str, *, prompt_dir: str | None = None) -> str:
    rel = Path(filename)

    if prompt_dir:
        candidate = Path(prompt_dir) / rel
        if not candidate.exists():
            raise FileNotFoundError(f"Translated prompt missing: {candidate}")
        return candidate.read_text(encoding="utf-8")

    for root in PROMPT_CANDIDATE_DIRS:
        candidate = root / rel
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Prompt not found: {filename}")

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


def _looks_like_direct_question(text: str) -> bool:
    t = (text or "").strip().lower()
    question_starts = (
        "ask ", "ask the ", "question ", "speak to ", "talk to ",
        "поцікав", "запит", "спит", "поговор", "розпит"
    )
    return t.endswith("?") or any(t.startswith(x) for x in question_starts)

def _looks_like_basic_read_or_review(text: str) -> bool:
    t = (text or "").strip().lower()
    markers = (
        "read ", "review ", "look at ", "check ", "inspect ", "examine ",
        "переглян", "прочит", "подив", "огля", "перевір", "вивчит"
    )
    return any(x in t for x in markers)

def _looks_like_basic_movement(text: str) -> bool:
    t = (text or "").strip().lower()
    markers = (
        "go to ", "enter ", "walk into ", "head to ", "move to ",
        "вируш", "увійт", "піти ", "прямув", "зайти ", "спустит"
    )
    return any(x in t for x in markers)

def classify_free_text_action(db: SessionDB, player_message: str) -> dict | None:
    text = (player_message or "").strip()
    lower = text.lower()
   
    # 1. Direct social questions to present NPCs -> automatic by default
    if _looks_like_direct_question(lower):
        return {
            "resolution_type": "automatic",
            "skill_name": None,
            "reason": "Direct question to a present NPC; no roll unless resistance is established."
        }

    # 2. Reading/reviewing visible material -> automatic unless hidden/technical
    if _looks_like_basic_read_or_review(lower):
        if any(x in lower for x in ["decode", "cipher", "hidden", "subtle", "pattern", "розшиф", "шифр", "прихован", "патерн"]):
            return {
                "resolution_type": "challenging",
                "skill_name": "Library Use",
                "reason": "This goes beyond obvious reading and requires deeper interpretation."
            }
        return {
            "resolution_type": "automatic",
            "skill_name": None,
            "reason": "Accessible records can be reviewed at the obvious level without a roll."
        }

    # 3. Movement into already discovered place -> automatic unless route hazard is established
    if _looks_like_basic_movement(lower):
        if any(x in lower for x in ["blizzard", "whiteout", "collapse", "unstable", "hazard", "метел", "обвал", "нестаб", "небезпеч"]):
            return {
                "resolution_type": "challenging",
                "skill_name": "Navigate",
                "reason": "Travel itself is hazardous or uncertain."
            }
        return {
            "resolution_type": "automatic",
            "skill_name": None,
            "reason": "Moving into an already discovered location does not require a roll by itself."
        }

    return None

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

def build_language_instruction(session_language: str) -> str:
    return get_language_name(session_language)


def is_roll_verdict_message(message: str) -> bool:
    text = message or ""
    return (
        "VERDICT" in text
        and ("[SYSTEM MESSAGE" in text or "ROLL_VERDICT:" in text)
    )


def has_roll_verdict(message: str) -> bool:
    return is_roll_verdict_message(message)


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


def assemble_keeper_prompt(
    *,
    include_roll_resolution: bool = False,
    include_scene_progression: bool = False,
    include_opening_scene: bool = False,
    prompt_dir: str | None = None,
) -> str:
    parts = [
        read_prompt("keeper/header.txt", prompt_dir=prompt_dir),
        read_prompt("keeper/output_contract.txt", prompt_dir=prompt_dir),
    ]
    if include_opening_scene:
        parts.append(read_prompt("keeper/opening_scene.txt", prompt_dir=prompt_dir))
    if include_scene_progression:
        parts.append(read_prompt("keeper/scene_progression.txt", prompt_dir=prompt_dir))
    parts.extend([
        read_prompt("keeper/action_adjudication.txt", prompt_dir=prompt_dir),
        read_prompt("keeper/core_identity.txt", prompt_dir=prompt_dir),
    ])
    if include_roll_resolution:
        parts.append(read_prompt("keeper/roll_resolution.txt", prompt_dir=prompt_dir))
    parts.append(
        "CAMPAIGN MODULE & CURRENT STATE\n{campaign_context}\n\nRECENT CHAT HISTORY\n{history}\n\nCURRENT PLAYER ACTION\n{action}"
    )
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) + "\n"


def extract_json(text: str) -> dict:
    raw_text = text or ""
    logger.info("extract_json(): raw response preview: %r", raw_text[:2000])

    text = raw_text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)

    tagged = re.search(
        r"<SYSTEM_RESPONSE_JSON>\s*(\{.*?\})\s*</SYSTEM_RESPONSE_JSON>",
        text,
        re.DOTALL,
    )
    if tagged:
        text = tagged.group(1).strip()
        logger.info("extract_json(): found SYSTEM_RESPONSE_JSON envelope")
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    logger.info("extract_json(): normalized candidate preview: %r", text[:2000])
    text = re.sub(r':\s*([-+]?\d+)[dD]\d+', r': \1', text)

    try:
        parsed = json.loads(text, strict=False)
        if isinstance(parsed, dict):
            parsed.setdefault("narrative", "")
            parsed.setdefault("suggested_actions", [])
            parsed.setdefault("state_updates", None)
            parsed.setdefault("combat_update", None)
            rr = parsed.get("roll_request")
            if not isinstance(rr, dict):
                rr = _detect_roll_request_from_suggested_actions(parsed.get("suggested_actions", [])) or _blank_roll_request()
            else:
                rr.setdefault("required", False)
                rr.setdefault("skill_name", "")
                rr.setdefault("action_text", "")
                rr.setdefault("reason", "")
                if not rr.get("required"):
                    derived = _detect_roll_request_from_suggested_actions(parsed.get("suggested_actions", []))
                    if derived:
                        rr = derived
            parsed["roll_request"] = rr
            return parsed
    except json.JSONDecodeError as e:
        logger.warning("extract_json(): JSON parse error: %s", e)
        logger.warning("extract_json(): failed candidate body: %r", text[:4000])

    narrative_match = re.search(
        r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    suggested_match = re.search(
        r'"suggested_actions"\s*:\s*\[(.*?)\]',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    narrative = ""
    if narrative_match:
        narrative = narrative_match.group(1)
        narrative = narrative.replace("\\n", "\n").replace('\\"', '"').strip()

    suggested_actions = []
    if suggested_match:
        raw_items = re.findall(r'"((?:[^"\\]|\\.)*)"', suggested_match.group(1), re.DOTALL)
        suggested_actions = [
            item.replace("\\n", " ").replace('\\"', '"').strip()
            for item in raw_items
            if item.strip()
        ][:3]

    action_block_match = re.search(
        r"^(.*?)(?:Suggested actions?)\s*:\s*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not narrative and action_block_match:
        narrative = action_block_match.group(1).strip()
        actions_blob = action_block_match.group(2).strip()
        action_lines = []
        for line in actions_blob.splitlines():
            clean = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line.strip())
            if clean:
                action_lines.append(clean)
        suggested_actions = action_lines[:3]

    if not narrative:
        narrative = text.strip()

    recovered = {
        "narrative": narrative,
        "suggested_actions": suggested_actions[:3] if suggested_actions else [],
        "roll_request": _detect_roll_request_from_suggested_actions(suggested_actions[:3] if suggested_actions else []) or _blank_roll_request(),
        "state_updates": {
            "character_name": "",
            "hp_change": 0,
            "sanity_change": 0,
            "mp_change": 0,
            "inventory_add": "",
            "inventory_remove": "",
            "location_name": "",
            "location_description": "",
            "clue_found": "",
            "clue_content": "",
            "thread_progress": "",
        },
    }
    logger.warning("extract_json(): recovered fallback payload: %r", recovered)
    return recovered


# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────

def get_chat_history(db: SessionDB, limit: int = 10) -> str:
    """
    Returns recent chat history for LLM context.
    Keeps a larger chronological window and lightly compresses long Keeper prose
    instead of collapsing history to only the last 4 messages.
    """
    events = db.list_events(limit=max(limit * 3, 24))
    pairs = []
    for e in events:
        if e.get("event_type") != "CHAT":
            continue
        payload = e.get("payload", {})
        role = payload.get("role", "Unknown")
        content = (payload.get("content", "") or "").strip()
        if not content:
            continue
        pairs.append((role, content))

    recent_pairs = pairs[-max(6, min(limit, 12)):]

    chat_lines = []
    for role, content in recent_pairs:
        role_norm = str(role).strip().lower()
        if role_norm == "keeper":
            if len(content) > 450:
                head = content[:180].strip()
                tail = content[-180:].strip()
                content = f"{head} … {tail}"
            chat_lines.append(f"KEEPER: {content}")
        else:
            if len(content) > 280:
                content = content[:280].strip()
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


def kv_set(db: SessionDB, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        (key, value),
    )
    db.conn.commit()


def _blank_roll_request() -> dict:
    return {"required": False, "skill_name": "", "action_text": "", "reason": ""}


def _detect_roll_request_from_suggested_actions(suggested_actions: list[str] | None) -> dict | None:
    for action in suggested_actions or []:
        m = re.match(r"^(.*?)\s*→\s*Roll\s+(.+?)\s*$", (action or '').strip(), re.IGNORECASE)
        if not m:
            continue
        return {
            "required": True,
            "action_text": m.group(1).strip(),
            "skill_name": m.group(2).strip(),
            "reason": "This action was already established as requiring a roll.",
        }
    return None


def load_pending_roll(db: SessionDB) -> dict | None:
    cur = db.conn.cursor()
    raw = kv_get(cur, "pending_roll_json", "")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("required"):
            return parsed
    except Exception:
        pass
    return None


def save_pending_roll(db: SessionDB, roll_request: dict | None) -> None:
    if not roll_request or not roll_request.get("required"):
        clear_pending_roll(db)
        return
    kv_set(db, "pending_roll_json", json.dumps(roll_request, ensure_ascii=False))


def clear_pending_roll(db: SessionDB) -> None:
    kv_set(db, "pending_roll_json", "")


def _tokenize_action_text(text: str) -> list[str]:
    text = (text or '').lower()
    tokens = re.findall(r"[\w']+", text, flags=re.UNICODE)
    stop = {"the","a","an","to","and","or","of","in","on","at","with","for","this","that","it","is","are","be","try","attempt"}
    return [t for t in tokens if len(t) > 2 and t not in stop]


def _is_same_intent(player_text: str, pending: dict) -> bool:
    player = set(_tokenize_action_text(player_text))
    pending_tokens = set(_tokenize_action_text((pending or {}).get("action_text", "")))
    if not player or not pending_tokens:
        return False
    overlap = len(player & pending_tokens)
    if overlap >= 2:
        return True
    if pending.get("skill_name") and pending["skill_name"].lower() in (player_text or '').lower():
        return True
    return False


def _classify_player_action(db: SessionDB, player_text: str) -> dict | None:
    text = (player_text or '').strip()
    if not text:
        return None
    
    heuristic = classify_free_text_action(db, text)
    if heuristic is not None:
        return {
            "resolution_type": heuristic.get("resolution_type", "automatic"),
            "skill_name": heuristic.get("skill_name", "") or "",
            "reason": heuristic.get("reason", "") or "",
        }
    
    cur = db.conn.cursor()
    current_objective = kv_get(cur, "current_objective", "")
    current_scene = kv_get(cur, "current_scene", "")
    current_location = kv_get(cur, "current_scene_location", "")
    pack = db.build_prompt_state_pack(limit_events=10)
    skills = []
    for pc in db.list_actors("PC"):
        for skill_name in (pc.get("skills") or {}).keys():
            if skill_name not in skills:
                skills.append(skill_name)
    last_keeper = ''
    for e in reversed(db.list_events(limit=8)):
        if e.get('event_type') == 'CHAT' and e.get('payload', {}).get('role') == 'Keeper':
            last_keeper = e.get('payload', {}).get('content', '')
            break

    prompt = (
        "Classify the player's typed action for a tabletop RPG.\n"
        "Return ONLY JSON with keys: resolution_type, skill_name, reason.\n"
        "resolution_type must be one of: automatic, challenging, impossible.\n"
        "Use challenging when the action is uncertain, risky, pressured, requires expertise, or was clearly framed as a discovery task.\n"
        "Use impossible only if an already-established barrier makes it impossible right now.\n"
        "If challenging, choose the best exact skill name from the list below.\n\n"
        f"CURRENT OBJECTIVE: {current_objective}\n"
        f"CURRENT SCENE: {current_scene}\n"
        f"CURRENT LOCATION: {current_location}\n"
        f"LAST KEEPER MESSAGE: {last_keeper[:700]}\n"
        f"AVAILABLE SKILLS: {', '.join(skills)}\n"
        f"PLAYER ACTION: {text}\n"
    )
    try:
        raw = get_llm(temperature=0.1).invoke(prompt)
        parsed = extract_json(str(raw)) if ('{' in str(raw) and '}' in str(raw)) else None
        if isinstance(parsed, dict):
            rt = str(parsed.get('resolution_type', '')).strip().lower()
            if rt in {'automatic','challenging','impossible'}:
                return {
                    'resolution_type': rt,
                    'skill_name': str(parsed.get('skill_name', '') or '').strip(),
                    'reason': str(parsed.get('reason', '') or '').strip(),
                }
    except Exception as e:
        logger.warning("action classifier failed: %s", e)

    # heuristic fallback
    low = text.lower()
    challenging_markers = ['decode','decrypt','translate','analyze','inspect','examine','search','расшиф','дешиф','огля','дослід','вивч','розшиф','символ']
    if any(m in low for m in challenging_markers):
        guess = ''
        for candidate in ['Spot Hidden', 'Library Use', 'Occult', 'History', 'Computer Use']:
            if candidate in skills:
                guess = candidate
                break
        return {'resolution_type': 'challenging', 'skill_name': guess, 'reason': 'The action requires careful interpretation or investigation under uncertainty.'}
    return {'resolution_type': 'automatic', 'skill_name': '', 'reason': ''}


def intercept_player_action_for_roll_gate(db: SessionDB, player_text: str) -> dict | None:
    text = (player_text or '').strip()
    lower = text.lower()

    if not text or text.startswith('/') or is_roll_verdict_message(text):
        return None
    if lower.startswith("[system"):
        return None
    if "start the story" in lower or "open with the first playable scene" in lower:
        return None
    
    pending = load_pending_roll(db)
    if pending and (_is_same_intent(text, pending) or not pending.get('action_text')):
        return {
            'narrative': 'This action still requires a roll before the outcome can be resolved.',
            'suggested_actions': [f"{pending.get('action_text') or 'Resolve the declared action'} → Roll {pending.get('skill_name') or 'Skill'}"],
            'state_updates': None,
            'roll_request': {
                'required': True,
                'skill_name': pending.get('skill_name', ''),
                'action_text': pending.get('action_text', ''),
                'reason': pending.get('reason', '') or 'A pending skill check is still unresolved.',
            },
            'image_url': None,
            'generation_id': None,
        }

    classified = _classify_player_action(db, text)
    if not classified:
        return None
    if classified.get('resolution_type') == 'challenging' and classified.get('skill_name'):
        rr = {
            'required': True,
            'skill_name': classified['skill_name'],
            'action_text': text,
            'reason': classified.get('reason', '') or 'This declared action requires a skill check.',
        }
        save_pending_roll(db, rr)
        return {
            'narrative': 'This declared action is not resolved automatically. Roll first, then the outcome can be narrated.',
            'suggested_actions': [f"{text} → Roll {classified['skill_name']}"],
            'state_updates': None,
            'roll_request': rr,
            'image_url': None,
            'generation_id': None,
        }
    if classified.get('resolution_type') == 'impossible':
        return {
            'narrative': classified.get('reason') or 'That cannot be done yet because of an established barrier.',
            'suggested_actions': [],
            'state_updates': None,
            'roll_request': _blank_roll_request(),
            'image_url': None,
            'generation_id': None,
        }
    return None


# ─────────────────────────────────────────────
# Verdict guard (injected into campaign_context on dice results)
# ─────────────────────────────────────────────

def build_verdict_guard(message: str) -> str:
    """
    Returns an anti-repetition instruction block when an English-only roll verdict
    control message is detected, empty string otherwise.
    """
    if is_roll_verdict_message(message):
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


def _safe_json_loads(value: str | None):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _filter_met_npcs(npcs_text: str, all_keeper_chat: str) -> str:
    lines = [line for line in (npcs_text or "").splitlines() if line.strip()]
    if not lines:
        return "(none yet)"

    met_lines = []
    lowered_chat = (all_keeper_chat or "").lower()
    for line in lines:
        words = [w for w in re.findall(r"[\w'-]+", line) if len(w) > 4]
        if any(word.lower() in lowered_chat for word in words):
            met_lines.append(line)

    return "\n".join(met_lines) if met_lines else "(none yet)"


def compact_blueprint_text(blueprint: dict | None) -> str:
    if not isinstance(blueprint, dict) or not blueprint:
        return "(none)"

    lines = []
    for key in ["title", "era_and_setting", "inciting_hook", "core_mystery", "hidden_threat", "atmosphere_notes"]:
        value = blueprint.get(key)
        if value:
            lines.append(f"{key}: {value}")

    acts = blueprint.get("acts") or []
    if acts:
        lines.append("acts:")
        for act in acts[:6]:
            act_no = act.get("act", "?")
            act_title = act.get("title", "")
            act_summary = act.get("summary", "")
            lines.append(f"- act={act_no}; title={act_title}; summary={act_summary}")
            for scene in (act.get("scenes") or [])[:4]:
                scene_name = scene.get("scene", "")
                location = scene.get("location", "")
                trigger = scene.get("trigger", "")
                what_happens = scene.get("what_happens", "")
                threat_level = scene.get("threat_level", "")
                lines.append(
                    f"  - scene={scene_name}; location={location}; trigger={trigger}; "
                    f"what_happens={what_happens}; threat_level={threat_level}"
                )

    for list_key in ["locations", "npcs", "clues", "plot_threads"]:
        items = blueprint.get(list_key) or []
        if not items:
            continue
        lines.append(f"{list_key}:")
        for item in items[:6]:
            if isinstance(item, dict):
                parts = []
                for field in ["name", "title", "role", "location", "stakes", "description", "content", "secret"]:
                    val = item.get(field)
                    if val:
                        parts.append(f"{field}={val}")
                if parts:
                    lines.append("- " + "; ".join(parts[:4]))
            else:
                lines.append(f"- {item}")

    return "\n".join(lines) if lines else "(none)"


def build_authoritative_context(db: SessionDB, *, campaign_atoms: str = "", themes: str = "STANDARD") -> tuple[str, str]:
    cur = db.conn.cursor()
    digest = kv_get(cur, "story_digest", "")
    blueprint_raw = kv_get(cur, "scenario_blueprint_json", "{}")
    current_objective = kv_get(cur, "current_objective", "")
    scenario_setting = kv_get(cur, "scenario_setting", "")
    era_context = kv_get(cur, "era_context", "")
    current_act = kv_get(cur, "current_act", "1")
    current_scene = kv_get(cur, "current_scene", "")
    current_scene_location = kv_get(cur, "current_scene_location", "")
    blueprint = _safe_json_loads(blueprint_raw)

    context_parts = []

    if digest:
        context_parts.append(
            "═══════════════════════════════════════════\n"
            "STORY SO FAR — READ THIS FIRST\n"
            "This is a factual continuity record. Treat it as ground truth.\n"
            "Do NOT contradict, repeat, or rediscover anything already established here.\n"
            f"{digest}\n"
            "═══════════════════════════════════════════"
        )

    if campaign_atoms:
        context_parts.append(
            f"--- SCENARIO ATOMS (THEMES: {themes}) ---\n"
            "This is the scenario anchor. Your next response must stay inside it.\n"
            f"{campaign_atoms}\n"
            "------------------------------------------"
        )

    if blueprint:
        context_parts.append(
            "--- AUTHORITATIVE SCENARIO BLUEPRINT ---\n"
            "This structured scenario state is more authoritative than your genre priors.\n"
            "Do not invent unrelated threats, locations, factions, or act transitions.\n"
            f"{compact_blueprint_text(blueprint)}\n"
            "----------------------------------------"
        )

        current_act_num = int(current_act or 1)
        hidden_threat = str(blueprint.get("hidden_threat", "") or "").strip()
        core_mystery = str(blueprint.get("core_mystery", "") or "").strip()

        if current_act_num <= 1:
            context_parts.append(
                "--- DISCLOSURE GATE ---\n"
                f"Do not fully reveal hidden_threat ({hidden_threat}) in Act 1.\n"
                f"Do not fully explain core_mystery ({core_mystery}) in Act 1.\n"
                "In Act 1, provide signs, rumors, traces, witnesses, and partial contradictions only.\n"
                "Named endgame entities, final mechanism logic, and full ritual purpose should emerge later unless directly forced by extraordinary player success.\n"
                "----------------------"
            )

    context_parts.append(
        "--- CURRENT ACT GATE ---\n"
        f"current_act: {current_act}\n"
        f"current_scene: {current_scene}\n"
        f"current_scene_location: {current_scene_location}\n"
        "Do not jump to later acts or reveal later-act truths unless the current scene has been resolved in play.\n"
        "------------------------"
    )

    context_parts.append(
        "--- ABANDONMENT RULE ---\n"
        "If the investigators retreat, leave, or try to abandon the current situation, do not simply relocate them to a neutral filler scene.\n"
        "Instead, advance the threat, worsen the stakes, create fallout, or show what consequence follows from walking away.\n"
        "Leaving the scene is allowed, but it must cost time, safety, or strategic position.\n"
        "------------------------"
    )

    if current_objective:
        context_parts.append(
            "--- CURRENT OBJECTIVE ---\n"
            f"{current_objective}\n"
            "Advance, replace, or complicate this objective — do not ignore it.\n"
            "-------------------------"
        )

    pack = db.build_prompt_state_pack(limit_events=12)
    all_keeper_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=80)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    )
    met_npcs = _filter_met_npcs(pack.get("npcs_text", ""), all_keeper_chat)

    state_str = (
        "=== AUTHORITATIVE GAME STATE ===\n"
        "The following is the current source-of-truth state from the session database.\n"
        "Respect it. Do not contradict it.\n\n"
        f"SCENARIO SETTING:\n{scenario_setting}\n\n"
        f"ERA CONTEXT:\n{era_context}\n\n"
        f"INVESTIGATORS:\n{pack.get('investigators_text')}\n\n"
        "MET NPCs (only these have been introduced on-screen; do not introduce hidden blueprint NPCs by name without cause):\n"
        f"{met_npcs}\n\n"
        f"CURRENT LOCATION:\n{pack.get('location_text')}\n\n"
        f"PLOT THREADS:\n{pack.get('threads_text')}\n\n"
        f"DISCOVERED CLUES:\n{pack.get('clues_text')}\n\n"
        f"RECENT MEANINGFUL EVENTS:\n{pack.get('recent_events_text')}"
    )

    return "\n\n".join(part for part in context_parts if part.strip()), state_str

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


# ─────────────────────────────────────────────
# LLM factory — provider-agnostic
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.7, *, streaming: bool = False, num_ctx: int | None = None):
    """
    Returns a LangChain LLM configured from .env.

    .env keys:
        LLM_PROVIDER   = "ollama" | "openai"          (default: ollama)
        OLLAMA_MODEL   = "gemma3:27b"                  (default)
        OLLAMA_BASE_URL= "http://localhost:11434"      (default)
        OPENAI_MODEL   = "gpt-4o"                      (default)
        OPENAI_API_KEY = "sk-..."                      (required for openai)

    num_ctx is forwarded to Ollama when supplied.
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

    kwargs = dict(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
    if num_ctx is not None:
        kwargs["num_ctx"] = int(num_ctx)

    return OllamaLLM(**kwargs)

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
                cur_hp = target.get("hp")
                cur_san = target.get("san")
                cur_mp = target.get("mp")

                new_hp = max(0, cur_hp + hp_change) if cur_hp is not None else None
                new_san = max(0, cur_san + san_change) if cur_san is not None else None
                new_mp = max(0, cur_mp + mp_change) if cur_mp is not None else None

                status = target.get("status", "ok")

                # Only promote to dead/insane when that stat is actually tracked.
                if cur_hp is not None and hp_change != 0 and new_hp == 0:
                    status = "dead"
                elif cur_san is not None and san_change != 0 and new_san == 0:
                    status = "insane"
                elif cur_hp is not None and hp_change < 0:
                    status = "injured"

                patch_kwargs = {
                    "actor_id": target["id"],
                    "status": status,
                }
                if new_hp is not None:
                    patch_kwargs["hp"] = new_hp
                if new_san is not None:
                    patch_kwargs["san"] = new_san
                if new_mp is not None:
                    patch_kwargs["mp"] = new_mp

                db.patch_actor(**patch_kwargs)

                db.log_event("STATE_UPDATE", {
                    "actor": target["name"],
                    "status": status,
                    **({"hp": f"{hp_change:+}→{new_hp}"} if hp_change and new_hp is not None else {}),
                    **({"san": f"{san_change:+}→{new_san}"} if san_change and new_san is not None else {}),
                    **({"mp": f"{mp_change:+}→{new_mp}"} if mp_change and new_mp is not None else {}),
                })

                result["updated_actor"] = {
                    "name": target["name"],
                    "status": status,
                    **({"hp": new_hp} if new_hp is not None else {}),
                    **({"san": new_san} if new_san is not None else {}),
                    **({"mp": new_mp} if new_mp is not None else {}),
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
            name=new_location,
            description=state_updates.get("location_description", "")
        )
        for pc in db.list_actors("PC"):
            db.patch_actor(actor_id=pc["id"], location_id=loc_id)
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

    # --- Objective evolution ---
    new_objective = ""

    if clue_found := state_updates.get("clue_found", ""):
        new_objective = f"Use what was learned from clue: {clue_found}"

    if new_location := state_updates.get("location_name", ""):
        new_objective = f"Act on what is now possible in {new_location}"

    if thread_name := state_updates.get("thread_progress", ""):
        new_objective = f"Push thread '{thread_name}' to its next concrete payoff"

    if new_objective:
        db.conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            ("current_objective", new_objective),
        )
        db.conn.commit()
