import json
import logging
import re
#pylint: disable=import-error
from utils.db_session import SessionDB
from utils.helpers import extract_json, get_llm, kv_get, kv_set, is_roll_verdict_message

logger = logging.getLogger("keeper_ai.helpers.actions")


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
