import json
import logging
import re

from utils.coc_skills import (
    CHARACTERISTIC_TARGETS,
    CORE_SKILL_BASES,
    SKILL_GLOSS,
    all_core_roll_targets,
    base_skill_value,
    normalize_roll_target_name,
)

# pylint: disable=import-error
from utils.db_session import SessionDB
from utils.helpers import kv_get, kv_set, is_roll_verdict_message
from utils.local_models import infer_effort_level_async, semantic_skill_shortlist_async
from utils.prompt_translate import normalize_language_code, translate_player_action_to_english

logger = logging.getLogger("keeper_ai.helpers.actions")

# CHARACTERISTIC_TARGETS = ("STR", "CON", "SIZ", "DEX", "APP", "INT", "POW", "EDU")

SOCIAL_ROLL_TARGETS = (
    "Psychology",
    "Persuade",
    "Fast Talk",
    "Charm",
    "Intimidate",
)

SOCIAL_PRESSURE_TARGETS = (
    "Intimidate",
    "Fast Talk",
    "Psychology",
    "Persuade",
    "Charm",
)

SOCIAL_GENTLE_TARGETS = (
    "Persuade",
    "Charm",
    "Psychology",
    "Fast Talk",
    "Intimidate",
)

TECHNICAL_ROLL_TARGETS = (
    "Electrical Repair",
    "Mechanical Repair",
    "Science",
    "Operate Heavy Machinery",
    "Computer Use",
    "Locksmith",
)

SYMBOLIC_ROLL_TARGETS = (
    "Language (Other)",
    "Occult",
    "Archaeology",
    "History",
    "Art/Craft",
    "Science",
)

PHYSICAL_ROLL_TARGETS = (
    "STR",
    "CON",
    "DEX",
    "POW",
    "Climb",
    "Jump",
    "Swim",
    "Dodge",
    "Stealth",
)


def _get_session_language(db: SessionDB) -> str:
    cur = db.conn.cursor()
    return normalize_language_code(kv_get(cur, "language", "en"))


def _resolve_player_action_to_canonical(db: SessionDB, player_text: str) -> str:
    """
    Resolve a clicked/displayed suggested action to the canonical action stored in DB.

    This function performs only a cheap exact map lookup. It never calls an LLM.
    """
    raw = str(player_text or "").strip()
    if not raw:
        return ""

    cur = db.conn.cursor()
    raw_map = kv_get(cur, "last_suggested_action_map_json", "")
    if not raw_map:
        return raw

    try:
        parsed = json.loads(raw_map)
    except Exception:
        return raw

    if not isinstance(parsed, dict):
        return raw

    mapped = str(parsed.get(raw, "") or "").strip()
    if mapped:
        logger.info("ACTION_CANONICALIZED_FROM_MAP raw=%r canonical=%r", raw, mapped)
        return mapped

    return raw


def _resolve_player_action_to_canonical_english(db: SessionDB, player_text: str) -> str:
    """
    Resolve player action for internal adjudication.

    Cost rule:
    - clicked suggested action: exact DB map lookup only, 0 GPT calls
    - English session/free text: no GPT call
    - non-English free text: one action-to-English translation call
    """
    raw = str(player_text or "").strip()
    if not raw:
        return ""

    resolved = _resolve_player_action_to_canonical(db, raw)
    if resolved != raw:
        return resolved

    lang = _get_session_language(db)
    if lang == "en":
        return resolved

    canonical_english = translate_player_action_to_english(
        resolved,
        source_language=lang,
    ).strip()

    if canonical_english and canonical_english != resolved:
        logger.info(
            "ACTION_CANONICALIZED_TO_ENGLISH raw=%r resolved=%r english=%r",
            raw,
            resolved,
            canonical_english,
        )

    return canonical_english or resolved


def _collect_available_skills(db: SessionDB) -> list[str]:
    """
    Return every canonical CoC skill as available.

    A missing skill on the actor sheet means the investigator uses its base value,
    not that the skill cannot be rolled.
    """
    skills: list[str] = list(CORE_SKILL_BASES.keys())

    # Preserve any custom sheet skills too.
    for pc in db.list_actors("PC"):
        for skill_name in (pc.get("skills") or {}).keys():
            skill = normalize_roll_target_name(skill_name)
            if skill and skill not in skills:
                skills.append(skill)

    return skills


def _collect_available_roll_targets(db: SessionDB) -> list[str]:
    targets = list(_collect_available_skills(db))
    for stat in CHARACTERISTIC_TARGETS:
        if stat not in targets:
            targets.append(stat)
    return targets


def _best_party_value_for_roll_target(db: SessionDB, target_name: str) -> int:
    target = normalize_roll_target_name(target_name)
    if not target:
        return 0

    best = 0
    target_upper = target.upper()

    for pc in db.list_actors("PC"):
        stats = pc.get("stats") or {}

        if target_upper in CHARACTERISTIC_TARGETS:
            value = stats.get(target_upper)
            if value is None:
                value = pc.get(target_upper.lower())
            best = max(best, int(value or 0))
            continue

        sheet_value = None
        for name, value in (pc.get("skills") or {}).items():
            if normalize_roll_target_name(name).lower() == target.lower():
                sheet_value = int(value or 0)
                break

        if sheet_value is None:
            sheet_value = base_skill_value(target, stats=stats)

        best = max(best, int(sheet_value or 0))

    # No PCs? Still return static base.
    if best <= 0 and target in CORE_SKILL_BASES:
        best = base_skill_value(target)

    return best


def _fallback_shortlist(available_targets: list[str]) -> list[str]:
    preferred = [
        "Spot Hidden",
        "Listen",
        "Library Use",
        "Psychology",
        "Persuade",
        "Fast Talk",
        "Charm",
        "Intimidate",

        "Locksmith",
        "Mechanical Repair",
        "Electrical Repair",
        "Computer Use",
        "Operate Heavy Machinery",

        "Appraise",
        "Art/Craft",
        "Accounting",
        "Law",
        "Credit Rating",

        "History",
        "Anthropology",
        "Archaeology",
        "Occult",
        "Cthulhu Mythos",
        "Language (Own)",
        "Language (Other)",
        "Science",

        "Medicine",
        "First Aid",
        "Psychoanalysis",
        "Natural World",
        "Survival",

        "Stealth",
        "Sleight of Hand",
        "Disguise",
        "Track",
        "Navigate",

        "Climb",
        "Jump",
        "Swim",
        "Throw",
        "Dodge",
        "Fighting (Brawl)",
        "Firearms (Handgun)",
        "Firearms (Rifle/Shotgun)",
        "Drive Auto",
        "Pilot",
        "Ride",

        "STR",
        "CON",
        "SIZ",
        "DEX",
        "APP",
        "INT",
        "POW",
        "EDU",
    ]

    picked = [s for s in preferred if s in available_targets][:8]
    return picked if picked else available_targets[:8]


def _token_set(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "into", "from", "that", "this",
        "more", "closely", "carefully", "try", "attempt", "about", "what",
        "who", "where", "when", "why", "how", "right", "now", "immediately",
        "object", "thing", "item", "something",
    }
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z()/-]{2,}", str(text or "").lower())
        if t not in stop
    }


def _score_roll_target_for_action(action_text: str, target: str) -> float:
    action_lower = str(action_text or "").lower()
    action_tokens = _token_set(action_lower)
    target_text = f"{target} {SKILL_GLOSS.get(target, target)}"
    target_tokens = _token_set(target_text)

    score = float(len(action_tokens & target_tokens))

    target_lower = target.lower()
    if re.search(r"(?<![a-z])" + re.escape(target_lower) + r"(?![a-z])", action_lower):
        score += 3.0

    anchors = {
        "Appraise": (
            "appraise", "value", "worth", "quality", "material",
            "craftsmanship", "authenticity", "identify object",
            "examine the object", "inspect the object",
        ),        
        "Listen": (
            "listen", "hear", "overhear", "eavesdrop",
            "recording", "audio", "voice", "distress call",
            "static", "transmission", "radio call", "signal audio",
        ),
        "Spot Hidden": (
            "spot", "notice", "search", "look for", "inspect visually",
            "scan the room", "binoculars", "closer look", "look closely",
            "examine", "inspect", "study", "check closely",
            "dark shape", "visual detail", "visible anomaly",
        ),
        "Track": (
            "track", "trail", "trace movement", "follow tracks",
        ),
        "Navigate": (
            "navigate", "route", "bearing", "course", "map", "charts",
            "pinpoint", "locate source", "triangulate", "source of the signal",
            "signal source", "underwater signal", "boat", "motorboat",
            "island", "reef", "current", "currents", "storm", "approach",
            "depart", "sail", "head for", "reach beacon island",
        ),
        "Pilot": (
            "pilot", "boat", "motorboat", "helm", "steer", "sail",
            "maneuver", "approach", "depart", "rough sea", "storm",
            "head for the island", "navigate the boat",
        ),
        "Drive Auto": (
            "drive", "car", "truck", "vehicle", "road", "steering",
        ),
        "Science": (
            "analyze", "scientific", "measurement", "scan", "sonar",
            "signal", "frequency", "interference", "source", "pattern",
            "structure", "metal", "alloy", "geometry", "anomaly",
            "underwater", "beam", "resonance",
        ),
        "Electrical Repair": (
            "radio", "frequency", "transmission", "transmitter",
            "receiver", "signal", "static", "interference", "sonar",
            "calibrate", "contact", "hail", "power", "circuit",
            "wiring", "beam", "lighthouse signal",
        ),
        "Mechanical Repair": (
            "machine", "mechanism", "gears", "pipes", "engine",
            "motor", "boat engine", "winch", "repair",
        ),
        "Computer Use": (
            "computer", "terminal", "digital", "software", "data",
            "logs", "metadata", "system",
        ),
        "Library Use": (
            "records", "archives", "research", "files", "ledger",
            "manifest", "logs", "radio logs", "previous communications",
            "old reports", "documents",
        ),
        "Locksmith": (
            "lock", "locked", "safe", "latch", "key", "keys", "pick",
            "pick the lock", "pick lock", "lockpick", "unlock", "disable the lock",
            "bypass the lock", "jimmy the lock", "hidden door",
        ),        
        "History": (
            "history", "historical", "old legend", "local legend",
            "past events", "old records", "decade", "tradition",
        ),
        "Occult": (
            "ritual", "occult", "cult", "supernatural", "myth",
            "forbidden rite", "esoteric",
        ),
        "Language (Other)": (
            "translate", "decipher", "foreign language", "unknown writing",
            "runes", "inscription", "glyphs",
        ),
        "Archaeology": (
            "artifact", "ceramic", "pottery", "ancient", "ruin",
            "excavation", "burial", "relic",
        ),
        "Psychology": (
            "read motive", "lying", "emotion", "nervous", "fear",
            "hesitate", "read him", "read her", "body language",
        ),
        "Persuade": (
            "convince", "persuade", "reason with", "negotiate", "appeal",
        ),
        "Fast Talk": (
            "bluff", "lie", "trick", "quick excuse", "fast talk",
        ),
        "Charm": (
            "charm", "befriend", "reassure", "rapport",
        ),
        "Intimidate": (
            "demand", "threaten", "intimidate", "coerce", "pressure",
            "confront",
        ),
        "Stealth": (
            "sneak", "hide", "quietly", "unseen",
        ),
        "Dodge": (
            "dodge", "avoid attack", "evade",
        ),
        "STR": (
            "force", "force open", "break", "break open", "push", "pull", "hold",
            "lift", "pry", "pry open", "kick open", "smash", "heave",
            "gate", "door", "hatch", "trapdoor", "grate", "bars", "rubble",
        ),
        "CON": (
            "resist", "endure", "poison", "fatigue", "cold", "heat",
            "pain", "hold breath",
        ),
        "DEX": (
            "balance", "reflex", "quick", "precision", "careful hands",
            "steady hands", "maneuver carefully",
        ),
        "INT": (
            "deduce", "logic", "figure out", "connect facts",
            "make sense", "infer", "understand pattern",
        ),
        "POW": (
            "willpower", "resist mental", "psychic", "sanity",
        ),
        "APP": (
            "impress", "appearance", "first impression",
        ),
        "EDU": (
            "remember", "know", "academic", "education",
        ),
    }

    for phrase in anchors.get(target, ()):
        if phrase in action_lower:
            score += 4.0

    return score

_PHYSICAL_OBJECT_RE = re.compile(
    r"\b("
    r"gate|door|hatch|trapdoor|grate|bars|barrier|panel|cover|lid|"
    r"cabinet|drawer|crate|lock|latch|lever|wheel|valve|shutter|"
    r"stone|boulder|rubble|debris|obstruction|beam|chain|rope"
    r")\b",
    flags=re.IGNORECASE,
)

_PHYSICAL_FORCE_VERB_RE = re.compile(
    r"\b("
    r"force|force open|break|break open|push|pull|lift|hold|"
    r"shove|pry|pry open|kick open|smash|heave|bend|overpower"
    r")\b",
    flags=re.IGNORECASE,
)

_SOCIAL_TARGET_RE = re.compile(
    r"\b("
    r"him|her|them|man|woman|guard|officer|worker|keeper|witness|"
    r"miller|davies|npc|person|people|cultist|attendant|operator"
    r")\b",
    flags=re.IGNORECASE,
)


def _is_physical_force_action(text: str) -> bool:
    text = str(text or "").strip().lower()
    if not text:
        return False

    if _PHYSICAL_FORCE_VERB_RE.search(text) and _PHYSICAL_OBJECT_RE.search(text):
        return True

    # Common compact forms.
    if re.search(r"\b(force|break|pry|kick|smash)\s+(it|this|that)\s+open\b", text):
        return True

    return False


def _is_social_pressure_action(text: str) -> bool:
    text = str(text or "").strip().lower()
    if not text:
        return False

    # Explicit social pressure verbs are social only when aimed at a person.
    if re.search(r"\b(interrogate|pressure|demand|threaten|confront|coerce|lean on|order)\b", text):
        return True

    # "force" is ambiguous. Treat it as social only if it targets a person/social compliance.
    if re.search(r"\b(force|push|press)\b", text) and _SOCIAL_TARGET_RE.search(text):
        return True

    if re.search(r"\b(make him|make her|make them|make the guard|make the officer)\b", text):
        return True

    return False

def _available_in_order(names: tuple[str, ...], available_targets: list[str]) -> list[str]:
    return [name for name in names if name in available_targets]


def _action_domain_pool(*, action_text: str, effort_level: str, available_targets: list[str]) -> list[str]:
    text = str(action_text or "").strip().lower()
    effort = str(effort_level or "").strip().lower()

    # IMPORTANT: physical force must be checked before social pressure.
    # "Force the gate" = STR. "Force the guard to talk" = Intimidate/Fast Talk/etc.
    if _is_physical_force_action(text):
        return _available_in_order(("STR", "Fighting (Brawl)", "Mechanical Repair", "Locksmith"), available_targets)

    if _is_social_pressure_action(text):
        return _available_in_order(SOCIAL_PRESSURE_TARGETS, available_targets)
    if re.search(
        r"\b("
        r"pick\s+(?:the\s+)?lock|"
        r"lockpick|"
        r"unlock|"
        r"open\s+(?:the\s+)?lock|"
        r"disable\s+(?:the\s+)?lock|"
        r"bypass\s+(?:the\s+)?lock|"
        r"jimmy\s+(?:the\s+)?lock|"
        r"locked\s+(?:door|gate|hatch|safe|cabinet|drawer)"
        r")\b",
        text,
    ):
        return _available_in_order(
            ("Locksmith", "Mechanical Repair", "DEX", "STR"),
            available_targets,
        )    
    if re.search(r"\b(question|ask|talk to|speak to|interview)\b", text):
        if effort == "opposed":
            return _available_in_order(SOCIAL_ROLL_TARGETS, available_targets)
        return _available_in_order(SOCIAL_GENTLE_TARGETS, available_targets)

    if re.search(r"\b(listen|hear|overhear|eavesdrop)\b", text):
        return _available_in_order(("Listen", "Stealth", "Psychology"), available_targets)

    if re.search(r"\b(search|look for|spot|notice|observe|scan the room|inspect visually)\b", text):
        return _available_in_order(("Spot Hidden", "Track", "Listen"), available_targets)

    if re.search(r"\b(sneak|hide|move quietly|stay unseen|avoid being seen)\b", text):
        return _available_in_order(("Stealth", "DEX"), available_targets)

    if re.search(r"\b(device|mechanism|machine|wiring|circuit|radio|transmitter|resonator|disable|repair)\b", text):
        return _available_in_order(TECHNICAL_ROLL_TARGETS, available_targets)

    if re.search(r"\b(decipher|decode|translate|rune|runes|symbol|symbols|glyph|glyphs|inscription|unknown writing)\b", text):
        return _available_in_order(SYMBOLIC_ROLL_TARGETS, available_targets)

    if re.search(r"\b(analyze|compare|cross-reference|determine|identify pattern|connect facts|make sense)\b", text):
        return _available_in_order(("Library Use", "Spot Hidden", "INT", "Science", "History"), available_targets)

    # Generic fallback for force with no explicit object.
    if re.search(r"\b(force|break|lift|push|pull|hold|wrestle|overpower)\b", text):
        return _available_in_order(("STR", "Fighting (Brawl)"), available_targets)

    if re.search(r"\b(balance|leap|jump|climb|crawl|dodge|evade|run)\b", text):
        return _available_in_order(PHYSICAL_ROLL_TARGETS, available_targets)

    if re.search(r"\b(resist|endure|hold breath|poison|fatigue|cold|heat|pain)\b", text):
        return _available_in_order(("CON", "POW"), available_targets)
    
    if re.search(r"\b(examine|inspect|study|look closely|closer look|look over|check closely)\b", text):
        # Generic close examination of a visible object.
        # Do not let MiniLM choose random physical skills like Throw just because "object" appears.
        return _available_in_order(
            (
                "Spot Hidden",
                "Appraise",
                "Art/Craft",
                "Science",
                "Archaeology",
                "Occult",
                "Mechanical Repair",
            ),
            available_targets,
        )
    return []


def _rank_roll_targets_for_action(
    db: SessionDB,
    *,
    action_text: str,
    candidate_targets: list[str],
) -> tuple[list[str], str, float]:
    """
    Rank roll targets by action meaning first, party value only as tie-breaker.

    Important:
    - Never pick high History just because the investigator is good at History.
    - If every semantic score is zero, preserve candidate order.
    """
    pool = [t for t in candidate_targets if t]
    if not pool:
        return [], "", 0.0

    scored = []
    for idx, target in enumerate(pool):
        semantic_score = _score_roll_target_for_action(action_text, target)
        party_value = _best_party_value_for_roll_target(db, target)
        scored.append((target, semantic_score, party_value, idx))

    best_score = max(score for _, score, _, _ in scored)

    if best_score <= 0:
        ranked = pool
        return ranked, ranked[0], 0.0

    scored.sort(
        key=lambda row: (
            row[1],          # semantic score
            row[2] / 100.0,  # party value only tie-breaks
            -row[3],         # preserve original order
        ),
        reverse=True,
    )

    ranked = [target for target, _, _, _ in scored]
    return ranked, ranked[0], float(scored[0][1])

def _get_last_keeper_message(db: SessionDB, limit: int = 8) -> str:
    for event in reversed(db.list_events(limit=limit)):
        if event.get("event_type") == "CHAT" and event.get("payload", {}).get("role") == "Keeper":
            return str(event.get("payload", {}).get("content", "") or "")
    return ""


def _extract_explicit_skill_request(player_text: str, available_skills: list[str]) -> str:
    text = str(player_text or "").strip().lower()
    if not text:
        return ""

    explicit_roll_pattern = re.search(
        r"\b(roll|skill check|make a roll|make a check)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not explicit_roll_pattern:
        return ""

    for skill in available_skills:
        skill_norm = str(skill or "").strip().lower()
        if not skill_norm:
            continue
        pattern = r"(?<![a-z])" + re.escape(skill_norm) + r"(?![a-z])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return skill

    return ""


def _forced_effort_from_canonical_action(player_text: str, last_keeper: str) -> dict | None:
    text = str(player_text or "").strip().lower()
    keeper = str(last_keeper or "").strip().lower()

    if not text:
        return None

    if any(marker in text for marker in ("roll ", "skill check", "make a check", "make a roll")):
        return {
            "effort_level": "expert",
            "reason": "The player explicitly asked for a skill-based check.",
        }
    lockpick_markers = (
        "pick the lock",
        "pick lock",
        "lockpick",
        "open the lock",
        "unlock the door",
        "unlock the gate",
        "unlock the hatch",
        "disable the lock",
        "bypass the lock",
        "jimmy the lock",
    )
    if any(marker in text for marker in lockpick_markers):
        return {
            "effort_level": "expert",
            "reason": "The action attempts to bypass or pick a lock.",
        }
    expert_markers = (
        "decipher", "decode", "interpret", "translate", "analyze",
        "determine what it means", "determine the purpose", "figure out what it does",
        "diagnose", "scan for meaning", "read the symbol", "read the glyph",
        "read the runes", "understand the symbol", "identify the pattern",
        "compare the pattern", "find hidden meaning", "look for hidden meaning",
        "look for anomalies", "check for anomalies", "trace the source",
        "reconstruct what happened", "cross-reference", "compare the dates",
        "compare the markings", "compare the handwriting",
    )
    if any(marker in text for marker in expert_markers):
        return {
            "effort_level": "expert",
            "reason": "The action asks for interpretation, decoding, technical purpose, or specialist inference.",
        }

    examine_markers = (
        "examine", "inspect", "study", "investigate", "look closely",
        "check the device", "check the mechanism", "take apart", "open the device",
        "test the device", "compare", "cross-reference", "side by side",
        "review the logs", "review the records",
    )
    hidden_or_expert_context = (
        "device", "mechanism", "machine", "prototype", "wiring", "servo",
        "battery", "circuit", "panel", "engine", "resonance", "anomaly",
        "static", "ozone", "modified", "rearranged", "malfunction",
        "impossible", "unusual", "not currently on the market", "coded",
        "encrypted", "damaged", "degraded", "redacted", "inconsistent",
        "contradiction", "pattern", "synchronized", "nearly identical",
        "substitution", "metadata", "logs", "surveillance", "blind spot",
        "permit denial", "telehealth", "symbol", "symbols", "glyph", "glyphs",
        "rune", "runes", "archaic", "unknown notation", "resonator",
        "manifest", "ledger", "label", "handwriting", "markings", "transfer",
    )
    if any(x in text for x in examine_markers) and any(x in keeper for x in hidden_or_expert_context):
        return {
            "effort_level": "expert",
            "reason": "The action seeks non-obvious information from a hidden, technical, symbolic, degraded, or pattern-based clue.",
        }

    opposed_markers = (
        "press ", "confront ", "interrogate ", "force ", "demand ", "threaten ",
        "lean on ", "push harder", "insist", "order ", "make him tell",
        "make her tell", "see if he is lying", "see if she is lying",
        "read him", "read her", "catch him lying", "catch her lying",
    )
    if any(marker in text for marker in opposed_markers):
        return {
            "effort_level": "opposed",
            "reason": "The action pressures, manipulates, or tests a resisting person.",
        }

    resistant_context = (
        "evasive", "withholding", "lying", "avoids eye contact", "nervous", "fear",
        "defensive", "hostile", "refuses", "won't answer", "asks you not to",
        "delete", "do not post", "do not discuss", "operator", "administrator",
        "office", "restricted", "rehearsed", "too quickly", "blocks access",
    )
    if any(x in keeper for x in resistant_context):
        if any(x in text for x in ("ask ", "question ", "press ", "confront ", "interrogate ", "talk to ", "demand ", "order ")):
            return {
                "effort_level": "opposed",
                "reason": "The target or institution is already presented as evasive, restrictive, or withholding.",
            }

    return None


async def _classify_effort_level(db: SessionDB, canonical_english_action: str) -> dict:
    text = str(canonical_english_action or "").strip()
    if not text:
        return {"effort_level": "routine", "reason": ""}

    cur = db.conn.cursor()
    current_objective = kv_get(cur, "current_objective", "")
    current_scene = kv_get(cur, "current_scene", "")
    current_location = kv_get(cur, "current_scene_location", "")
    last_keeper = _get_last_keeper_message(db)

    forced = _forced_effort_from_canonical_action(text, last_keeper)
    if forced:
        return forced

    return await infer_effort_level_async(
        player_text=text,
        current_objective=current_objective,
        current_scene=current_scene,
        current_location=current_location,
        last_keeper=last_keeper,
    )


async def _shortlist_candidate_targets(
    *,
    player_text: str,
    current_objective: str,
    current_scene: str,
    current_location: str,
    last_keeper: str,
    available_targets: list[str],
) -> list[str]:
    if not available_targets:
        return []

    try:
        ranked = await semantic_skill_shortlist_async(
            player_text=player_text,
            current_objective=current_objective,
            current_scene=current_scene,
            current_location=current_location,
            last_keeper=last_keeper,
            available_skills=available_targets,
            top_k=6,
        )
        if ranked:
            return [x for x in ranked[:6] if x in available_targets]
    except Exception as e:
        logger.warning("local MiniLM shortlist failed: %s", e)

    return _fallback_shortlist(available_targets)


async def _classify_player_action(db: SessionDB, canonical_english_action: str) -> dict | None:
    text = str(canonical_english_action or "").strip()
    if not text:
        return None

    cur = db.conn.cursor()
    current_objective = kv_get(cur, "current_objective", "")
    current_scene = kv_get(cur, "current_scene", "")
    current_location = kv_get(cur, "current_scene_location", "")
    last_keeper = _get_last_keeper_message(db)

    available_skills = _collect_available_skills(db)
    roll_targets = _collect_available_roll_targets(db)
    if not roll_targets:
        return {"resolution_type": "automatic", "skill_name": "", "reason": ""}

    explicit_skill = _extract_explicit_skill_request(text, available_skills)
    if explicit_skill:
        return {
            "resolution_type": "challenging",
            "skill_name": explicit_skill,
            "reason": f"The player explicitly requested a {explicit_skill} check.",
            "candidate_skills": [explicit_skill],
            "effort_level": "expert",
        }

    effort = await _classify_effort_level(db, text)
    effort_level = str(effort.get("effort_level", "careful") or "careful").lower()
    effort_reason = str(effort.get("reason", "") or "")

    if effort_level in {"routine", "careful"}:
        return {
            "resolution_type": "automatic",
            "skill_name": "",
            "reason": effort_reason,
        }

    if effort_level == "impossible":
        return {
            "resolution_type": "impossible",
            "skill_name": "",
            "reason": effort_reason or "An established barrier prevents this action right now.",
        }

    domain_pool = _action_domain_pool(
        action_text=text,
        effort_level=effort_level,
        available_targets=roll_targets,
    )

    if domain_pool:
        selection_pool = domain_pool
        candidate_targets = domain_pool
    else:
        candidate_targets = await _shortlist_candidate_targets(
            player_text=text,
            current_objective=current_objective,
            current_scene=current_scene,
            current_location=current_location,
            last_keeper=last_keeper,
            available_targets=roll_targets,
        )
        selection_pool = candidate_targets

    if not selection_pool:
        return {
            "resolution_type": "automatic",
            "skill_name": "",
            "reason": "No valid roll target matched this action.",
            "candidate_skills": candidate_targets,
            "effort_level": effort_level,
        }

    ranked_candidates, selected_skill, top_score = _rank_roll_targets_for_action(
        db,
        action_text=text,
        candidate_targets=selection_pool,
    )

    logger.info(
        "ROLL_GATE_TARGET_PICK action=%r effort=%r domain_pool=%s candidates=%s ranked=%s selected=%r top_score=%.2f",
        text,
        effort_level,
        domain_pool,
        candidate_targets,
        ranked_candidates,
        selected_skill,
        top_score,
    )

    # If there was no domain-specific match and no semantic signal, do not invent a roll
    # from the investigator's highest skill.
    if not domain_pool and top_score < 2.0:
        return {
            "resolution_type": "automatic",
            "skill_name": "",
            "reason": "No clean roll target matched this action strongly enough.",
            "candidate_skills": ranked_candidates,
            "effort_level": effort_level,
        }

    if not selected_skill:
        return {
            "resolution_type": "automatic",
            "skill_name": "",
            "reason": "No valid roll target matched this action.",
            "candidate_skills": candidate_targets,
            "effort_level": effort_level,
        }

    return {
        "resolution_type": "challenging",
        "skill_name": selected_skill,
        "reason": effort_reason or f"The action is {effort_level} and needs a skill check.",
        "candidate_skills": ranked_candidates,
        "effort_level": effort_level,
    }


def _blank_roll_request() -> dict:
    return {"required": False, "skill_name": "", "action_text": "", "reason": ""}


def _detect_roll_request_from_suggested_actions(suggested_actions: list[str] | None) -> dict | None:
    # Compatibility export for current engine_chat.py. Move this to utils.helpers in the next file cleanup.
    for action in suggested_actions or []:
        m = re.match(r"^(.*?)\s*→\s*Roll\s+(.+?)\s*$", str(action or "").strip(), re.IGNORECASE)
        if not m:
            continue
        return {
            "required": True,
            "action_text": m.group(1).strip(),
            "skill_name": m.group(2).strip(),
            "reason": "This action was already established as requiring a roll.",
        }
    return None


def _empty_state_updates() -> dict:
    return {
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
    }


def _empty_combat_action() -> dict:
    return {
        "start_combat": False,
        "end_combat": False,
        "actor_name": "",
        "target_name": "",
        "action_type": "",
        "skill_name": "",
        "weapon_name": "",
        "weapon_damage": "",
        "defender_option": "",
        "shots_fired": 0,
        "bonus_dice": 0,
        "penalty_dice": 0,
    }


def _roll_gate_response(*, narrative: str, suggested_actions: list[str], roll_request: dict) -> dict:
    return {
        "narrative": narrative,
        "suggested_actions": suggested_actions,
        "state_updates": _empty_state_updates(),
        "combat_action": _empty_combat_action(),
        "roll_request": roll_request,
        "scene_entities": {"present_named_entities": []},
        "image_url": None,
        "generation_id": None,
    }


def load_pending_roll(db: SessionDB) -> dict | None:
    cur = db.conn.cursor()
    raw = kv_get(cur, "pending_roll_json", "")
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    if isinstance(parsed, dict) and parsed.get("required"):
        return parsed

    return None


def save_pending_roll(db: SessionDB, roll_request: dict | None) -> None:
    if not roll_request or not roll_request.get("required"):
        clear_pending_roll(db)
        return
    kv_set(db, "pending_roll_json", json.dumps(roll_request, ensure_ascii=False))


def clear_pending_roll(db: SessionDB) -> None:
    kv_set(db, "pending_roll_json", "")


def _tokenize_action_text(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "to", "and", "or", "of", "in", "on", "at", "with", "for",
        "this", "that", "it", "is", "are", "be", "try", "attempt",
    }
    tokens = re.findall(r"[\w']+", str(text or "").lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2 and t not in stop]


def _is_same_intent(player_text: str, pending: dict) -> bool:
    player = set(_tokenize_action_text(player_text))
    pending_tokens = set(_tokenize_action_text((pending or {}).get("action_text", "")))

    if not player or not pending_tokens:
        return False

    overlap = len(player & pending_tokens)
    if overlap >= 2:
        return True

    skill_name = str((pending or {}).get("skill_name", "") or "").lower()
    return bool(skill_name and skill_name in str(player_text or "").lower())


async def intercept_player_action_for_roll_gate(db: SessionDB, player_text: str) -> dict | None:
    raw_text = str(player_text or "").strip()
    lower = raw_text.lower()

    if not raw_text or raw_text.startswith("/") or is_roll_verdict_message(raw_text):
        return None
    if lower.startswith("[system"):
        return None
    if "start the story" in lower or "open with the first playable scene" in lower:
        return None

    canonical_text = _resolve_player_action_to_canonical_english(db, raw_text)

    pending = load_pending_roll(db)
    if pending and (
        _is_same_intent(canonical_text, pending)
        or _is_same_intent(raw_text, pending)
        or not pending.get("action_text")
    ):
        rr = {
            "required": True,
            "skill_name": pending.get("skill_name", ""),
            "action_text": pending.get("action_text", ""),
            "reason": pending.get("reason", "") or "A pending skill check is still unresolved.",
        }
        return _roll_gate_response(
            narrative="This action still requires a roll before the outcome can be resolved.",
            suggested_actions=[
                f"{pending.get('action_text') or 'Resolve the declared action'} → Roll {pending.get('skill_name') or 'Skill'}"
            ],
            roll_request=rr,
        )

    action_class = await _classify_player_action(db, canonical_text)
    logger.info(
        "ROLL_GATE_CLASSIFIED raw=%r canonical=%r class=%s",
        raw_text,
        canonical_text,
        json.dumps(action_class, ensure_ascii=False),
    )

    if not action_class:
        return None

    resolution_type = str(action_class.get("resolution_type", "automatic") or "automatic")
    if resolution_type != "challenging":
        logger.info(
            "ROLL_GATE_PASS_THROUGH raw=%r canonical=%r resolution_type=%r",
            raw_text,
            canonical_text,
            resolution_type,
        )
        clear_pending_roll(db)
        return None

    skill_name = str(action_class.get("skill_name", "") or "").strip()
    reason = str(action_class.get("reason", "") or "").strip()
    if not skill_name:
        logger.info("ROLL_GATE_NO_SKILL raw=%r canonical=%r class=%s", raw_text, canonical_text, action_class)
        clear_pending_roll(db)
        return None

    rr = {
        "required": True,
        "skill_name": skill_name,
        "action_text": canonical_text,
        "reason": reason,
    }
    save_pending_roll(db, rr)

    logger.info(
        "ROLL_GATE_TRIGGERED raw=%r canonical=%r skill=%r reason=%r",
        raw_text,
        canonical_text,
        skill_name,
        reason,
    )

    return _roll_gate_response(
        narrative="This action has meaningful uncertainty, so a roll is needed before the outcome is narrated.",
        suggested_actions=[f"{canonical_text} → Roll {skill_name}"],
        roll_request=rr,
    )
