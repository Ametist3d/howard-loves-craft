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
    "hr": "Croatian",    
    "ru": "Russian",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
}

def normalize_language_code(lang: str | None) -> str:
    raw = (lang or "en").strip().lower()
    return raw if raw else "en"

def get_language_name(lang: str | None) -> str:
    code = normalize_language_code(lang)
    return LANGUAGE_NAMES.get(code, code)

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

def _translate_user_facing_text(text: str, language: str) -> str:
    language = normalize_language_code(language)
    if not text or language == "en":
        return text

    llm = get_llm(temperature=0.1)
    target_language = get_language_name(language)

    prompt = (
        f"Translate the following game text into {target_language}.\n"
        "Rules:\n"
        "- Preserve names, identifiers, and formatting.\n"
        "- Do not add commentary.\n"
        "- Output only the translation.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        translated = llm.invoke(prompt).strip()
        return translated or text
    except Exception:
        return text
    
def localize_opening_result_fields(result: dict, language: str) -> dict:
    language = normalize_language_code(language)
    if language == "en":
        return result

    safe = dict(result or {})
    safe["narrative"] = _translate_user_facing_text(safe.get("narrative", ""), language)

    actions = safe.get("suggested_actions") or []
    safe["suggested_actions"] = [
        _translate_user_facing_text(str(a), language) for a in actions
    ]

    updates = dict(safe.get("state_updates") or {})
    if updates.get("location_description"):
        updates["location_description"] = _translate_user_facing_text(updates["location_description"], language)
    if updates.get("clue_content"):
        updates["clue_content"] = _translate_user_facing_text(updates["clue_content"], language)
    if updates.get("thread_progress"):
        updates["thread_progress"] = _translate_user_facing_text(updates["thread_progress"], language)
    safe["state_updates"] = updates

    return safe

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

def _match_registry_items_by_names(items: list[dict], names: list[str], key: str = "name") -> list[dict]:
    wanted = {_normalize_name(x) for x in (names or []) if str(x).strip()}
    out = []
    for item in items or []:
        value = _normalize_name(str(item.get(key, "") or ""))
        if value and value in wanted:
            out.append(item)
    return out

def _compact_registry_slice_for_scene(blueprint: dict, current_scene_obj: dict) -> str:
    if not isinstance(blueprint, dict):
        return ""

    lines = []

    current_location = str(current_scene_obj.get("location", "") or "").strip()
    current_npcs = [str(x) for x in (current_scene_obj.get("npc_present") or []) if x]
    current_clues = [str(x) for x in (current_scene_obj.get("clues_available") or []) if x]

    locations = blueprint.get("locations") or []
    npcs = blueprint.get("npcs") or []
    clues = blueprint.get("clues") or []
    threads = blueprint.get("plot_threads") or []

    matched_locations = _match_registry_items_by_names(locations, [current_location], key="name")
    matched_npcs = _match_registry_items_by_names(npcs, current_npcs, key="name")
    matched_clues = _match_registry_items_by_names(clues, current_clues, key="title")

    if matched_locations:
        lines.append("CURRENT LOCATION REGISTRY:")
        for loc in matched_locations:
            lines.append(
                f"- {loc.get('name','')}"
                f" | tags={loc.get('tags','')}"
                f" | hidden={loc.get('hidden','')}"
            )

    if matched_npcs:
        lines.append("CURRENT NPC REGISTRY:")
        for npc in matched_npcs:
            lines.append(
                f"- {npc.get('name','')}"
                f" | role={npc.get('role','')}"
                f" | secret={npc.get('secret','')}"
                f" | motivation={npc.get('motivation','')}"
            )

    if matched_clues:
        lines.append("CURRENT CLUE REGISTRY:")
        for clue in matched_clues:
            lines.append(
                f"- {clue.get('title','')}"
                f" | content={clue.get('content','')}"
                f" | true_meaning={clue.get('true_meaning','')}"
                f" | location={clue.get('location','')}"
            )

    if threads:
        lines.append("ACTIVE PLOT THREADS:")
        for th in threads[:2]:
            lines.append(
                f"- {th.get('name','')}"
                f" | stakes={th.get('stakes','')}"
                f" | steps={th.get('steps','')}"
            )

    return "\n".join(lines)

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

def extract_blueprint_json(text: str) -> dict:
    raw_text = (text or "").strip()
    logger.info("extract_blueprint_json(): raw preview: %r", raw_text[:4000])

    cleaned = raw_text

    # Remove markdown fences
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)

    # Keep only the largest JSON-looking object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No complete JSON object found in blueprint response")

    candidate = cleaned[start:end + 1]

    # --- soft repair layer for common model artifacts ---
    # remove obvious template junk
    candidate = candidate.replace("<%", "")
    candidate = candidate.replace("%>", "")

    # normalize a few observed broken keys / artifacts
    candidate = re.sub(
        r'"dynamic_press[^"]*pressures"\s*:',
        '"dynamic_pressures":',
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r'"exit_[a-zA-Z]+_condition"\s*:',
        '"exit_condition":',
        candidate,
        flags=re.IGNORECASE,
    )

    # remove accidental control characters
    candidate = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", candidate)

    logger.info("extract_blueprint_json(): candidate preview: %r", candidate[:4000])

    parsed = json.loads(candidate, strict=False)

    if not isinstance(parsed, dict):
        raise ValueError("Blueprint response is not a JSON object")

    required = ["title", "era_and_setting", "inciting_hook", "acts"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"Blueprint missing required keys: {missing}")

    if not isinstance(parsed.get("acts"), list) or not parsed["acts"]:
        raise ValueError("Blueprint acts must be a non-empty list")

    if not str(parsed.get("title", "")).strip():
        raise ValueError("Blueprint title is empty")

    if not str(parsed.get("inciting_hook", "")).strip():
        raise ValueError("Blueprint inciting_hook is empty")

    return parsed

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


def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _get_current_scene_obj(blueprint: dict, current_act: str, current_scene: str) -> dict:
    if not isinstance(blueprint, dict):
        return {}

    try:
        act_num = int(current_act or 1)
    except Exception:
        act_num = 1

    for act in blueprint.get("acts") or []:
        if int(act.get("act", 0) or 0) != act_num:
            continue
        for scene in act.get("scenes") or []:
            if _normalize_name(scene.get("scene", "")) == _normalize_name(current_scene):
                return scene
    return {}


def _get_next_scene_obj(blueprint: dict, current_act: str, current_scene: str) -> dict:
    if not isinstance(blueprint, dict):
        return {}

    try:
        act_num = int(current_act or 1)
    except Exception:
        act_num = 1

    acts = blueprint.get("acts") or []
    for act_idx, act in enumerate(acts):
        if int(act.get("act", 0) or 0) != act_num:
            continue

        scenes = act.get("scenes") or []
        for i, scene in enumerate(scenes):
            if _normalize_name(scene.get("scene", "")) == _normalize_name(current_scene):
                if i + 1 < len(scenes):
                    return scenes[i + 1]
                if act_idx + 1 < len(acts):
                    next_act_scenes = acts[act_idx + 1].get("scenes") or []
                    return next_act_scenes[0] if next_act_scenes else {}
    return {}


def compact_blueprint_text(
    blueprint: dict | None,
    *,
    current_act: str = "1",
    current_scene: str = "",
    met_npc_names: list[str] | None = None,
    found_clue_titles: list[str] | None = None,
) -> str:
    """
    Return ONLY the playable slice of the blueprint.
    Do not expose hidden_threat, core_mystery, late acts, unseen NPC secrets, or hidden clues.
    """
    if not isinstance(blueprint, dict) or not blueprint:
        return "(none)"

    met_npc_names = {_normalize_name(x) for x in (met_npc_names or []) if x}
    found_clue_titles = {_normalize_name(x) for x in (found_clue_titles or []) if x}

    lines = []

    # Public/top-level only
    for key in ["title", "era_and_setting", "inciting_hook", "atmosphere_notes"]:
        value = blueprint.get(key)
        if value:
            lines.append(f"{key}: {value}")

    current_scene_obj = _get_current_scene_obj(blueprint, current_act, current_scene)
    next_scene_obj = _get_next_scene_obj(blueprint, current_act, current_scene)

    lines.append("current_progression:")
    lines.append(f"- current_act={current_act}")
    lines.append(f"- current_scene={current_scene}")

    if current_scene_obj:
        lines.append("current_scene_frame:")
        for field in [
            "scene", "location", "scene_function", "dramatic_question",
            "entry_condition", "exit_condition", "trigger",
            "description", "what_happens", "pressure_if_delayed", "threat_level"
        ]:
            value = current_scene_obj.get(field)
            if value:
                lines.append(f"- {field}={value}")

        visible_scene_npcs = current_scene_obj.get("npc_present") or []
        if visible_scene_npcs:
            lines.append("- scene_npcs=" + ", ".join(str(x) for x in visible_scene_npcs if x))

        visible_scene_clues = current_scene_obj.get("clues_available") or []
        if visible_scene_clues:
            lines.append("- scene_clues=" + ", ".join(str(x) for x in visible_scene_clues if x))

    if next_scene_obj:
        lines.append("next_reachable_scene_hint:")
        for field in ["scene", "location", "trigger", "threat_level"]:
            value = next_scene_obj.get(field)
            if value:
                lines.append(f"- {field}={value}")

    # Only met NPCs or NPCs explicitly present in current scene
    current_scene_npcs = {
        _normalize_name(x) for x in (current_scene_obj.get("npc_present") or []) if x
    }
    allowed_npcs = met_npc_names | current_scene_npcs
    npcs = blueprint.get("npcs") or []
    if allowed_npcs and npcs:
        lines.append("allowed_npcs:")
        for npc in npcs:
            npc_name = str(npc.get("name", "") or "")
            if _normalize_name(npc_name) not in allowed_npcs:
                continue
            parts = [f"name={npc_name}"]
            if npc.get("description"):
                parts.append(f"description={npc['description']}")
            if npc.get("role"):
                parts.append(f"role={npc['role']}")
            lines.append("- " + "; ".join(parts))

    # Only found clues + current scene clue titles, never full hidden clue set
    clues = blueprint.get("clues") or []
    current_scene_clue_titles = {
        _normalize_name(x) for x in (current_scene_obj.get("clues_available") or []) if x
    }
    allowed_clues = found_clue_titles | current_scene_clue_titles
    if clues and allowed_clues:
        lines.append("allowed_clues:")
        for clue in clues:
            title = str(clue.get("title", "") or "")
            if _normalize_name(title) not in allowed_clues:
                continue
            parts = [f"title={title}"]
            if clue.get("location"):
                parts.append(f"location={clue['location']}")
            # content only for already-found clues
            if _normalize_name(title) in found_clue_titles and clue.get("content"):
                parts.append(f"content={clue['content']}")
            lines.append("- " + "; ".join(parts))

    # Threads are okay to expose in summary form
    threads = blueprint.get("plot_threads") or []
    if threads:
        lines.append("plot_threads:")
        for item in threads[:6]:
            if not isinstance(item, dict):
                continue
            parts = []
            for field in ["name", "stakes"]:
                val = item.get(field)
                if val:
                    parts.append(f"{field}={val}")
            if parts:
                lines.append("- " + "; ".join(parts))

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

    pack = db.build_prompt_state_pack(limit_events=12)
    all_keeper_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=80)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    )
    met_npcs = _filter_met_npcs(pack.get("npcs_text", ""), all_keeper_chat)

    met_npc_names = []
    for line in met_npcs.splitlines():
        clean = line.strip().lstrip("-").strip()
        if not clean:
            continue
        met_npc_names.append(clean.split("—", 1)[0].strip())

    found_clue_titles = [c.get("title", "") for c in db.list_clues(status="found", limit=100)]

    if blueprint:
        context_parts.append(
            "--- AUTHORITATIVE SCENARIO BLUEPRINT ---\n"
            "This is the playable, current slice of the scenario blueprint.\n"
            "It is more authoritative than your genre priors.\n"
            "Do not invent unrelated threats, locations, factions, or act transitions.\n"
            f"{compact_blueprint_text(blueprint, current_act=current_act, current_scene=current_scene, met_npc_names=met_npc_names, found_clue_titles=found_clue_titles)}\n"
            "----------------------------------------"
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

def _current_state_snapshot_for_validation(db: SessionDB) -> dict:
    cur = db.conn.cursor()
    blueprint = _safe_json_loads(kv_get(cur, "scenario_blueprint_json", "{}"))
    current_act = kv_get(cur, "current_act", "1")
    current_scene = kv_get(cur, "current_scene", "")
    current_scene_location = kv_get(cur, "current_scene_location", "")

    found_clues = db.list_clues(status="found", limit=200)
    hidden_clues = db.list_clues(status="hidden", limit=200)

    all_keeper_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=120)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    )

    pack = db.build_prompt_state_pack(limit_events=12)
    met_npcs_blob = _filter_met_npcs(pack.get("npcs_text", ""), all_keeper_chat)

    met_npc_names = []
    for line in met_npcs_blob.splitlines():
        clean = line.strip().lstrip("-").strip()
        if not clean:
            continue
        met_npc_names.append(clean.split("—", 1)[0].strip())

    current_scene_obj = _get_current_scene_obj(blueprint, current_act, current_scene)

    return {
        "blueprint": blueprint,
        "current_act": current_act,
        "current_scene": current_scene,
        "current_scene_location": current_scene_location,
        "current_scene_obj": current_scene_obj,
        "met_npc_names": met_npc_names,
        "found_clue_titles": [c.get("title", "") for c in found_clues],
        "hidden_clue_titles": [c.get("title", "") for c in hidden_clues],
    }


def validate_llm_response_against_state(db: SessionDB, result: dict) -> list[str]:
    snap = _current_state_snapshot_for_validation(db)
    narrative = (result.get("narrative") or "")
    updates = result.get("state_updates") or {}
    suggested_actions = result.get("suggested_actions") or []
    suggestions_blob = "\n".join(str(x) for x in suggested_actions if x)
    blueprint = snap["blueprint"]
    current_scene_obj = snap["current_scene_obj"] or {}
    current_scene_clues = {
        _normalize_name(x) for x in (current_scene_obj.get("clues_available") or []) if x
    }

    violations = []

    # 1) Unintroduced NPC by name
    blueprint_npcs = blueprint.get("npcs") or []
    allowed_npcs = {_normalize_name(x) for x in snap["met_npc_names"]}
    allowed_npcs |= {_normalize_name(x) for x in (current_scene_obj.get("npc_present") or []) if x}

    for npc in blueprint_npcs:
        npc_name = str(npc.get("name", "") or "").strip()
        if not npc_name:
            continue
        if _normalize_name(npc_name) in allowed_npcs:
            continue
        if npc_name.lower() in narrative.lower():
            violations.append(f"unintroduced_npc:{npc_name}")

    # 2) Hidden clue mentioned as already known
    for clue_title in snap["hidden_clue_titles"]:
        norm_title = _normalize_name(clue_title)
        if not clue_title:
            continue

        # If this clue is explicitly available in the current scene,
        # mentioning it as an object/lead is allowed.
        if norm_title in current_scene_clues:
            continue

        if clue_title.lower() in narrative.lower():
            if norm_title != _normalize_name(str(updates.get("clue_found", "") or "")):
                violations.append(f"undiscovered_clue_revealed:{clue_title}")

    # 3) Later-act location jump
    try:
        current_act_num = int(snap["current_act"] or 1)
    except Exception:
        current_act_num = 1

    for act in blueprint.get("acts") or []:
        try:
            act_num = int(act.get("act", 0) or 0)
        except Exception:
            act_num = 0
        if act_num <= current_act_num:
            continue
        for scene in act.get("scenes") or []:
            loc = str(scene.get("location", "") or "").strip()
            if loc and (loc.lower() in narrative.lower() or loc.lower() in suggestions_blob.lower()):
                violations.append(f"later_act_location:{loc}")

    # 4) Location update to place not reachable from current slice
    next_scene_obj = _get_next_scene_obj(blueprint, snap["current_act"], snap["current_scene"])
    allowed_locations = {
        _normalize_name(snap["current_scene_location"]),
        _normalize_name(current_scene_obj.get("location", "")),
        _normalize_name(next_scene_obj.get("location", "")),
    }
    new_location = str(updates.get("location_name", "") or "").strip()
    if new_location and _normalize_name(new_location) not in allowed_locations:
        violations.append(f"unearned_location_jump:{new_location}")

    return violations

def looks_like_valid_keeper_response(raw_text: str, parsed: dict | None) -> bool:
    text = (raw_text or "").strip()

    if not isinstance(parsed, dict):
        return False

    required_keys = {"narrative", "suggested_actions", "roll_request"}
    if not required_keys.issubset(parsed.keys()):
        return False

    if "<SYSTEM_RESPONSE_JSON>" in text and "</SYSTEM_RESPONSE_JSON>" in text:
        return True

    return True

def validate_opening_scene_response(db: SessionDB, result: dict) -> list[str]:
    """
    Lighter validation for the very first playable scene.
    We allow entering the first scene location and introducing NPCs/clues that
    belong to that scene, but still block later-act jumps and premature truth reveals.
    """
    snap = _current_state_snapshot_for_validation(db)
    blueprint = snap["blueprint"]
    current_scene_obj = snap["current_scene_obj"] or {}

    narrative = (result.get("narrative") or "")
    updates = result.get("state_updates") or {}
    suggested_actions = result.get("suggested_actions") or []
    suggestions_blob = "\n".join(str(x) for x in suggested_actions if x)

    violations = []

    if not narrative.strip():
        violations.append("opening_missing_narrative")

    if not isinstance(suggested_actions, list) or len([x for x in suggested_actions if str(x).strip()]) < 2:
        violations.append("opening_missing_actions")

    if str(updates.get("thread_progress", "") or "").strip():
        violations.append("opening_thread_progress_too_early")

    # Allow ONLY the current scene location (or empty)
    start_location = _normalize_name(current_scene_obj.get("location", "") or snap["current_scene_location"])
    new_location = _normalize_name(str(updates.get("location_name", "") or ""))
    if new_location and start_location and new_location != start_location:
        violations.append(f"opening_wrong_location:{updates.get('location_name', '')}")

    # Do not reveal hidden truth / core mystery directly in opening
    hidden_threat = str(blueprint.get("hidden_threat", "") or "").strip()
    core_mystery = str(blueprint.get("core_mystery", "") or "").strip()
    never_suspect = str(blueprint.get("truth_the_players_never_suspect", "") or "").strip()

    lowered = (narrative + "\n" + suggestions_blob).lower()
    for forbidden, tag in (
        (hidden_threat, "opening_hidden_threat"),
        (core_mystery, "opening_core_mystery"),
        (never_suspect, "opening_final_truth"),
    ):
        if forbidden and forbidden.lower() in lowered:
            violations.append(tag)

    # Do not jump to later-act locations
    try:
        current_act_num = int(snap["current_act"] or 1)
    except Exception:
        current_act_num = 1

    for act in blueprint.get("acts") or []:
        try:
            act_num = int(act.get("act", 0) or 0)
        except Exception:
            act_num = 0
        if act_num <= current_act_num:
            continue
        for scene in act.get("scenes") or []:
            loc = str(scene.get("location", "") or "").strip()
            if loc and loc.lower() in lowered:
                violations.append(f"opening_later_act_location:{loc}")

    return violations

def build_opening_fallback_result(db: SessionDB) -> dict:
    """
    Deterministic fallback opening if the LLM opening fails.
    """
    cur = db.conn.cursor()
    blueprint = _safe_json_loads(kv_get(cur, "scenario_blueprint_json", "{}"))
    current_act = kv_get(cur, "current_act", "1")
    current_scene = kv_get(cur, "current_scene", "")
    current_scene_location = kv_get(cur, "current_scene_location", "")
    current_objective = kv_get(cur, "current_objective", "")

    scene_obj = _get_current_scene_obj(blueprint, current_act, current_scene)

    scene_name = str(scene_obj.get("scene", "") or current_scene)
    location = str(scene_obj.get("location", "") or current_scene_location or "the current scene")
    description = str(scene_obj.get("description", "") or "")
    what_happens = str(scene_obj.get("what_happens", "") or "")
    npcs = [str(x) for x in (scene_obj.get("npc_present") or []) if x]
    clues = [str(x) for x in (scene_obj.get("clues_available") or []) if x]

    narrative_parts = []
    narrative_parts.append(f"You arrive at {location}.")
    if description:
        narrative_parts.append(description)
    if what_happens:
        narrative_parts.append(what_happens)
    elif scene_name:
        narrative_parts.append(f"This is the opening scene: {scene_name}.")
    if current_objective:
        narrative_parts.append(f"Right now, your immediate objective is: {current_objective}")

    suggested_actions = []
    if npcs:
        suggested_actions.append(f"Talk to {npcs[0]}")
    if clues:
        suggested_actions.append(f"Examine {clues[0]}")
    suggested_actions.append(f"Survey {location} for anything immediately out of place")

    # keep exactly 3
    suggested_actions = suggested_actions[:3]
    while len(suggested_actions) < 3:
        suggested_actions.append("Press the most obvious lead in front of you")

    return {
        "narrative": " ".join(x.strip() for x in narrative_parts if x.strip()),
        "suggested_actions": suggested_actions,
        "state_updates": {
            "character_name": "",
            "hp_change": 0,
            "sanity_change": 0,
            "mp_change": 0,
            "inventory_add": "",
            "inventory_remove": "",
            "location_name": location,
            "location_description": description,
            "clue_found": "",
            "clue_content": "",
            "thread_progress": ""
        },
        "combat_update": {
            "combat_started": False,
            "combat_ended": False,
            "attacker_name": "",
            "target_name": "",
            "attack_mode": "",
            "weapon_name": "",
            "weapon_damage": "",
            "skill_name": "",
            "defender_option": "",
            "shots_fired": 0,
            "bonus_dice": 0,
            "penalty_dice": 0
        },
        "roll_request": {
            "required": False,
            "skill_name": "",
            "action_text": "",
            "reason": ""
        }
    }

def sanitize_llm_result_on_validation_failure(result: dict, violations: list[str]) -> dict:
    """
    Conservative server-side sanitization.
    Do not replace the whole response unless absolutely necessary.
    """
    safe = dict(result or {})
    safe.setdefault("narrative", "")
    safe.setdefault("suggested_actions", [])
    safe.setdefault("state_updates", {})
    safe.setdefault("roll_request", {"required": False, "skill_name": "", "action_text": "", "reason": ""})

    updates = dict(safe.get("state_updates") or {})

    # Hard violations that require stripping state changes
    hard_prefixes = (
        "unintroduced_npc:",
        "later_act_location:",
        "unearned_location_jump:",
    )

    # Soft violations: keep narrative, just prevent premature state mutation
    soft_prefixes = (
        "undiscovered_clue_revealed:",
    )

    has_hard = any(v.startswith(hard_prefixes) for v in violations)
    has_soft = any(v.startswith(soft_prefixes) for v in violations)

    if has_soft:
        updates["clue_found"] = ""
        updates["clue_content"] = ""
        safe["state_updates"] = updates

    if has_hard:
        updates["location_name"] = ""
        updates["location_description"] = ""
        updates["clue_found"] = ""
        updates["clue_content"] = ""
        updates["thread_progress"] = ""
        safe["state_updates"] = updates

        # Keep the model's narrative if possible, but strip forced roll gate
        safe["roll_request"] = {"required": False, "skill_name": "", "action_text": "", "reason": ""}

        # If there is no usable narrative left, use an English system-safe fallback
        if not (safe.get("narrative") or "").strip():
            safe["narrative"] = (
                "The situation remains tense, but no new confirmed breakthrough occurs yet. "
                "Press the current scene using only details already established on-screen."
            )

        if not safe.get("suggested_actions"):
            safe["suggested_actions"] = [
                "Examine a detail already present in the current scene",
                "Ask a direct question to someone already on-screen",
                "Test an already-established lead"
            ]

    return safe

def advance_blueprint_progress(db: SessionDB, result: dict) -> None:
    cur = db.conn.cursor()
    blueprint = _safe_json_loads(kv_get(cur, "scenario_blueprint_json", "{}"))
    if not isinstance(blueprint, dict) or not blueprint:
        return

    current_act = kv_get(cur, "current_act", "1")
    current_scene = kv_get(cur, "current_scene", "")
    current_scene_obj = _get_current_scene_obj(blueprint, current_act, current_scene)
    if not current_scene_obj:
        return

    updates = result.get("state_updates") or {}
    clue_found = str(updates.get("clue_found", "") or "").strip()
    location_name = str(updates.get("location_name", "") or "").strip()

    scene_clues = {_normalize_name(x) for x in (current_scene_obj.get("clues_available") or []) if x}
    scene_location = _normalize_name(current_scene_obj.get("location", ""))

    payoff_hit = False

    if clue_found and _normalize_name(clue_found) in scene_clues:
        payoff_hit = True

    if location_name and _normalize_name(location_name) != scene_location:
        payoff_hit = True

    if not payoff_hit:
        return

    acts = blueprint.get("acts") or []
    try:
        current_act_num = int(current_act or 1)
    except Exception:
        current_act_num = 1

    for act_idx, act in enumerate(acts):
        if int(act.get("act", 0) or 0) != current_act_num:
            continue
        scenes = act.get("scenes") or []
        for scene_idx, scene in enumerate(scenes):
            if _normalize_name(scene.get("scene", "")) != _normalize_name(current_scene):
                continue

            # next scene in same act
            if scene_idx + 1 < len(scenes):
                next_scene = scenes[scene_idx + 1]
                kv_set(db, "current_scene", str(next_scene.get("scene", "") or ""))
                kv_set(db, "current_scene_location", str(next_scene.get("location", "") or ""))
                return

            # first scene of next act
            if act_idx + 1 < len(acts):
                next_act = acts[act_idx + 1]
                next_scenes = next_act.get("scenes") or []
                kv_set(db, "current_act", str(next_act.get("act", current_act_num + 1)))
                if next_scenes:
                    kv_set(db, "current_scene", str(next_scenes[0].get("scene", "") or ""))
                    kv_set(db, "current_scene_location", str(next_scenes[0].get("location", "") or ""))
                return
            
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

    # --- Blueprint progression ---
    advance_blueprint_progress(db, result)
