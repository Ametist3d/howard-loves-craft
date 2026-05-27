import logging
import re

# pylint: disable=import-error
from utils.db_session import SessionDB
from utils.helpers import (
    _normalize_name,
    _safe_json_loads,
    kv_get,
    kv_set,
    read_prompt,
)

logger = logging.getLogger("keeper_ai.helpers.state")

_LOCATION_STOPWORDS = {
    "the", "a", "an", "of", "in", "at", "on",
    "room", "area", "sector", "zone", "site", "place",
    "north", "south", "east", "west",
}

_EMPTY_STATE_UPDATES = {
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

_EMPTY_COMBAT_ACTION = {
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

_EMPTY_ROLL_REQUEST = {
    "required": False,
    "skill_name": "",
    "action_text": "",
    "reason": "",
}


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        clean = _clean_text(item)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out

def _current_pc_names(db: SessionDB) -> list[str]:
    return [_clean_text(a.get("name", "")) for a in db.list_actors("PC") if _clean_text(a.get("name", ""))]


def _location_tokens(text: str) -> set[str]:
    norm = _normalize_name(text or "")
    words = re.findall(r"[a-z0-9]+", norm)
    return {w for w in words if len(w) >= 4 and w not in _LOCATION_STOPWORDS}


def _same_locationish(a: str, b: str) -> bool:
    na = _normalize_name(a or "")
    nb = _normalize_name(b or "")
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True

    ta = _location_tokens(a)
    tb = _location_tokens(b)
    if not ta or not tb:
        return False

    overlap = len(ta & tb)
    return overlap >= max(1, min(len(ta), len(tb)) // 2)


def _filter_met_npcs(npcs_text: str, all_keeper_chat: str) -> str:
    """
    Keep only NPC lines that have actually appeared in prior Keeper narration.
    Scene-local NPCs are handled separately by build_current_scene_prompt_pack().
    """
    lines = [line for line in (npcs_text or "").splitlines() if line.strip()]
    if not lines:
        return "(none yet)"

    lowered_chat = (all_keeper_chat or "").lower()
    met_lines: list[str] = []
    for line in lines:
        words = [w for w in re.findall(r"[\w'-]+", line) if len(w) > 4]
        if any(word.lower() in lowered_chat for word in words):
            met_lines.append(line)

    return "\n".join(met_lines) if met_lines else "(none yet)"


def _get_current_scene_obj(blueprint: dict, current_act: str, current_scene: str) -> dict:
    if not isinstance(blueprint, dict):
        return {}

    try:
        act_num = int(current_act or 1)
    except Exception:
        act_num = 1

    wanted_scene = _normalize_name(current_scene)
    for act in blueprint.get("acts") or []:
        try:
            if int(act.get("act", 0) or 0) != act_num:
                continue
        except Exception:
            continue

        for scene in act.get("scenes") or []:
            if _normalize_name(scene.get("scene", "")) == wanted_scene:
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
    wanted_scene = _normalize_name(current_scene)
    for act_idx, act in enumerate(acts):
        try:
            if int(act.get("act", 0) or 0) != act_num:
                continue
        except Exception:
            continue

        scenes = act.get("scenes") or []
        for scene_idx, scene in enumerate(scenes):
            if _normalize_name(scene.get("scene", "")) != wanted_scene:
                continue
            if scene_idx + 1 < len(scenes):
                return scenes[scene_idx + 1]
            if act_idx + 1 < len(acts):
                next_act_scenes = acts[act_idx + 1].get("scenes") or []
                return next_act_scenes[0] if next_act_scenes else {}
    return {}


def _story_scene_display_name(scene: dict | None) -> str:
    if not scene:
        return ""
    return _clean_text(scene.get("name") or scene.get("scene") or "")


def build_current_scene_prompt_pack(db: SessionDB) -> str:
    scene = db.get_current_story_scene()
    if not scene:
        return "(none)"

    payload = scene.get("payload") or {}
    npcs = db.list_story_scene_npcs(scene["id"])
    clues = db.list_story_scene_clues(scene["id"], include_hidden=True)
    threads = db.list_story_scene_threads(scene["id"])

    lines = [
        "CURRENT STORY NODE:",
        f"- act_no={scene.get('act_no')}",
        f"- scene_no={scene.get('scene_no')}",
        f"- scene={scene.get('name')}",
        f"- location={scene.get('location_name', '')}",
    ]

    for key in (
        "scene_function",
        "dramatic_question",
        "entry_condition",
        "exit_condition",
        "trigger",
        "description",
        "what_happens",
        "pressure_if_delayed",
        "threat_level",
        "keeper_notes",
    ):
        value = _clean_text(payload.get(key, ""))
        if value:
            lines.append(f"- {key}={value}")

    reveals = [_clean_text(x) for x in payload.get("reveals") or [] if _clean_text(x)]
    conceals = [_clean_text(x) for x in payload.get("conceals") or [] if _clean_text(x)]
    if reveals:
        lines.append("- reveals=" + " | ".join(reveals[:3]))
    if conceals:
        lines.append("- conceals=" + " | ".join(conceals[:3]))

    if npcs:
        lines.append("SCENE NPCS:")
        for npc in npcs[:6]:
            lines.append(f"- {npc['name']} | role={npc.get('description', '')}")

    if clues:
        lines.append("SCENE CLUES:")
        for clue in clues[:8]:
            lines.append(f"- {clue['title']}")

    if threads:
        lines.append("SCENE THREADS:")
        for thread in threads[:4]:
            lines.append(f"- {thread['name']} | stakes={thread.get('stakes', '')}")

    return "\n".join(lines)


def assemble_keeper_prompt(
    *,
    include_roll_resolution: bool = False,
    include_scene_progression: bool = False,
    include_opening_scene: bool = False,
    prompt_dir: str | None = None,
) -> str:
    """
    Assemble the runtime Keeper prompt.

    Runtime prompts are internal/canonical English. The selected user language is
    handled only after final JSON parsing by prompt_translate.py.
    """
    parts = [
        read_prompt("keeper/header.txt", prompt_dir=prompt_dir),
        read_prompt("keeper/output_contract.txt", prompt_dir=prompt_dir),
        (
            "INTERNAL LANGUAGE CONTRACT\n"
            "- Write the entire SYSTEM_RESPONSE_JSON in English.\n"
            "- Keep state_updates, roll_request, combat_action, clue titles, thread notes, and suggested_actions in English.\n"
            "- Do not switch to the player's display language inside the JSON.\n"
            "- User-facing translation happens after this JSON is parsed."
        ),
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
        "CAMPAIGN MODULE & CURRENT STATE\n{campaign_context}\n\n"
        "RECENT CHAT HISTORY\n{history}\n\n"
        "CURRENT PLAYER ACTION\n{action}"
    )
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) + "\n"


def _get_story_progression_from_db(db: SessionDB) -> tuple[str, str, str]:
    scene = db.get_current_story_scene()
    if scene:
        return (
            str(scene.get("act_no") or "1"),
            _story_scene_display_name(scene),
            _clean_text(scene.get("location_name", "")),
        )

    cur = db.conn.cursor()
    return (
        kv_get(cur, "current_act", "1"),
        kv_get(cur, "current_scene", ""),
        kv_get(cur, "current_scene_location", ""),
    )


def _current_state_npcs_and_chat(db: SessionDB) -> tuple[str, list[str]]:
    pack = db.build_prompt_state_pack(limit_events=12)
    all_keeper_chat = " ".join(
        e.get("payload", {}).get("content", "")
        for e in db.list_events(limit=80)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    )
    met_npcs_text = _filter_met_npcs(pack.get("npcs_text", ""), all_keeper_chat)

    names: list[str] = []
    for line in met_npcs_text.splitlines():
        clean = line.strip().lstrip("-").strip()
        if clean and clean != "(none yet)":
            names.append(clean.split("—", 1)[0].split("|", 1)[0].strip())
    return met_npcs_text, names


def build_authoritative_context(
    db: SessionDB,
    *,
    campaign_atoms: str = "",
    themes: str = "STANDARD",
    include_next_hint: bool = True,
) -> tuple[str, str]:
    """
    Build the live-play context.

    The session DB story graph is authoritative. Global scenario DB retrieval is
    intentionally not used here; live chat should not reintroduce unrelated or
    later-act scenario atoms.
    """
    cur = db.conn.cursor()
    digest = kv_get(cur, "story_digest", "")
    current_objective = kv_get(cur, "current_objective", "")
    scenario_setting = kv_get(cur, "scenario_setting", "")
    era_context = kv_get(cur, "era_context", "")
    current_act, current_scene, current_scene_location = _get_story_progression_from_db(db)

    context_parts: list[str] = [
        "--- INTERNAL CANONICAL MODE ---\n"
        "All runtime reasoning and SYSTEM_RESPONSE_JSON output must be English.\n"
        "The player's display language is handled only by final display translation.\n"
        "Do not localize names, skills, or state fields inside canonical JSON.\n"
        "--------------------------------"
    ]

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
            "Scenario anchor only. Do not expose hidden/future truths unless the current scene warrants it.\n"
            f"{campaign_atoms}\n"
            "------------------------------------------"
        )

    scene_pack = build_current_scene_prompt_pack(db)
    context_parts.append(
        "--- AUTHORITATIVE CURRENT SCENE ---\n"
        "This scene row from the session DB is the source of truth.\n"
        f"{scene_pack}\n"
        "-----------------------------------"
    )

    next_scene = None
    current_db_scene = db.get_current_story_scene()
    if include_next_hint and current_db_scene:
        next_scene = db.get_next_story_scene(current_db_scene["id"])

    context_parts.append(
        "--- CURRENT ACT GATE ---\n"
        f"current_act: {current_act}\n"
        f"current_scene: {current_scene}\n"
        f"current_scene_location: {current_scene_location}\n"
        "Do not jump to later acts or reveal later-act truths unless the current scene has been resolved in play.\n"
        "Only move to the next reachable scene when the player's action is movement/transition intent.\n"
        "------------------------"
    )

    if next_scene:
        context_parts.append(
            "--- NEXT REACHABLE SCENE, NOT CURRENTLY PRESENT ---\n"
            f"scene: {next_scene.get('name', '')}\n"
            f"location: {next_scene.get('location_name', '')}\n"
            "Use this only if the player explicitly moves, follows the lead, enters, travels, descends, or proceeds.\n"
            "---------------------------------------------------"
        )

    context_parts.append(
        "--- ABANDONMENT RULE ---\n"
        "If investigators retreat, leave, or abandon the current situation, do not relocate them to neutral filler.\n"
        "Advance the threat, worsen stakes, create fallout, or show consequences.\n"
        "------------------------"
    )

    if current_objective:
        context_parts.append(
            "--- CURRENT OBJECTIVE ---\n"
            f"{current_objective}\n"
            "Advance, replace, or complicate this objective; do not ignore it.\n"
            "-------------------------"
        )

    pack = db.build_prompt_state_pack(limit_events=12)
    met_npcs, _ = _current_state_npcs_and_chat(db)

    state_str = (
        "=== AUTHORITATIVE GAME STATE ===\n"
        "The following is current source-of-truth state from SQLite. Respect it.\n\n"
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


def maybe_force_movement_progress(db: SessionDB, player_action: str, result: dict) -> dict:
    """
    Mark movement/transition intent and bind that movement to the next DB story scene.

    This prevents the LLM from creating infinite transitional sublocations like:
    - newly opened niche
    - narrow passage
    - deeper tunnel
    - another chamber entrance

    The session DB story graph remains authoritative.
    """
    action = (player_action or "").strip().lower()
    if not action:
        return result

    movement_re = re.compile(
        r"\b("
        r"go(?:\s+to|\s+inside|\s+in|\s+down)?|"
        r"enter|"
        r"step\s+into|"
        r"walk\s+into|"
        r"head\s+to|"
        r"move(?:\s+to|\s+into)?|"
        r"continue(?:\s+forward)?|"
        r"keep\s+going|"
        r"keep\s+moving|"
        r"descend|"
        r"climb\s+down|"
        r"follow|"
        r"proceed|"
        r"travel\s+to|"
        r"crawl\s+through|"
        r"pass\s+through"
        r")\b",
        flags=re.IGNORECASE,
    )

    if not movement_re.search(action):
        return result

    safe = dict(result or {})
    safe["_player_movement_intent"] = True

    updates = {**_EMPTY_STATE_UPDATES, **dict(safe.get("state_updates") or {})}

    current_scene = db.get_current_story_scene()
    if not current_scene:
        safe["state_updates"] = updates
        return safe

    next_scene = db.get_next_story_scene(current_scene["id"])
    if not next_scene:
        safe["state_updates"] = updates
        return safe

    safe["_movement_from_story_scene_id"] = current_scene["id"]

    current_location = _clean_text(current_scene.get("location_name", ""))
    next_location = _clean_text(next_scene.get("location_name", ""))
    next_description = _clean_text(next_scene.get("location_description", ""))

    llm_location = _clean_text(updates.get("location_name", ""))

    if not next_location:
        safe["state_updates"] = updates
        return safe

    # If the LLM invented a transitional sub-location, override it with the
    # next authoritative story-scene location.
    should_bind_to_next_scene = (
        not llm_location
        or _same_locationish(llm_location, current_location)
        or not _same_locationish(llm_location, next_location)
    )

    if should_bind_to_next_scene:
        logger.info(
            "MOVEMENT_BIND_TO_NEXT_SCENE action=%r llm_location=%r current_scene=%r next_scene=%r next_location=%r",
            player_action,
            llm_location,
            current_scene.get("name", ""),
            next_scene.get("name", ""),
            next_location,
        )
        updates["location_name"] = next_location
        if next_description and not _clean_text(updates.get("location_description", "")):
            updates["location_description"] = next_description

    if _same_locationish(_clean_text(updates.get("location_name", "")), next_location):
        safe["_scene_advance_allowed"] = True

    safe["state_updates"] = updates
    return safe


def _clean_opening_objective_text(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"^\s*Act\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)

    for pat in (
        r"follow this lead:\s*(.+)$",
        r"establish what is happening here\s*[—\-]\s*(.+)$",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .")
    return text.strip(" .")


def _mentions_location_explicitly(location: str, *texts: str) -> bool:
    loc_norm = _normalize_name(location)
    if not loc_norm:
        return False
    for text in texts:
        if loc_norm in _normalize_name(text or ""):
            return True
    return False


def _current_state_snapshot_for_validation(db: SessionDB) -> dict:
    cur = db.conn.cursor()
    blueprint = _safe_json_loads(kv_get(cur, "scenario_blueprint_json", "{}"))

    db_scene = db.get_current_story_scene()
    if db_scene:
        current_act = str(db_scene.get("act_no") or "1")
        current_scene = _story_scene_display_name(db_scene)
        current_scene_location = _clean_text(db_scene.get("location_name", ""))
        current_scene_no = int(db_scene.get("scene_no", 0) or 0)
        current_scene_obj = db_scene.get("payload") or {}
        current_scene_npc_names = [
            _clean_text(n.get("name", ""))
            for n in db.list_story_scene_npcs(db_scene["id"])
            if _clean_text(n.get("name", ""))
        ]
        current_scene_clue_titles = [
            _clean_text(c.get("title", ""))
            for c in db.list_story_scene_clues(db_scene["id"], include_hidden=True)
            if _clean_text(c.get("title", ""))
        ]
    else:
        current_act = kv_get(cur, "current_act", "1")
        current_scene = kv_get(cur, "current_scene", "")
        current_scene_location = kv_get(cur, "current_scene_location", "")
        current_scene_no = 0
        current_scene_obj = _get_current_scene_obj(blueprint, current_act, current_scene)
        current_scene_npc_names = [_clean_text(x) for x in current_scene_obj.get("npc_present") or [] if _clean_text(x)]
        current_scene_clue_titles = [_clean_text(x) for x in current_scene_obj.get("clues_available") or [] if _clean_text(x)]

    found_clues = db.list_clues(status="found", limit=200)
    hidden_clues = db.list_clues(status="hidden", limit=200)
    met_npcs_blob, met_npc_names = _current_state_npcs_and_chat(db)

    return {
        "blueprint": blueprint or {},
        "current_act": current_act,
        "current_scene": current_scene,
        "current_scene_location": current_scene_location,
        "current_scene_no": current_scene_no,
        "current_scene_obj": current_scene_obj,
        "current_scene_npc_names": current_scene_npc_names,
        "current_scene_clue_titles": current_scene_clue_titles,
        "met_npc_names": met_npc_names,
        "met_npcs_blob": met_npcs_blob,
        "found_clue_titles": [_clean_text(c.get("title", "")) for c in found_clues if _clean_text(c.get("title", ""))],
        "hidden_clue_titles": [_clean_text(c.get("title", "")) for c in hidden_clues if _clean_text(c.get("title", ""))],
    }


def validate_llm_response_against_state(db: SessionDB, result: dict) -> list[str]:
    snap = _current_state_snapshot_for_validation(db)
    narrative = _clean_text(result.get("narrative") or "")
    updates = result.get("state_updates") or {}
    suggested_actions = result.get("suggested_actions") or []
    suggestions_blob = "\n".join(_clean_text(x) for x in suggested_actions if _clean_text(x))

    blueprint = snap["blueprint"] or {}
    current_scene_obj = snap["current_scene_obj"] or {}
    current_scene_clues = {_normalize_name(x) for x in snap.get("current_scene_clue_titles") or [] if x}
    violations: list[str] = []

    try:
        current_act_num = int(snap["current_act"] or 1)
    except Exception:
        current_act_num = 1
    current_scene_no = int(snap.get("current_scene_no") or 0)

    # Future-scene NPC mentions.
    future_scene_npcs: set[str] = set()
    for act in blueprint.get("acts") or []:
        try:
            act_num = int(act.get("act", 0) or 0)
        except Exception:
            act_num = 0
        for idx, scene in enumerate(act.get("scenes") or [], start=1):
            if not ((act_num > current_act_num) or (act_num == current_act_num and idx > current_scene_no)):
                continue
            # for npc_name in scene.get("npc_present") or []:
            #     clean = _clean_text(npc_name)
            #     if clean:
            #         future_scene_npcs.add(clean)

    combined_text = f"{narrative}\n{suggestions_blob}".lower()
    for npc_name in sorted(future_scene_npcs):
        if npc_name.lower() in combined_text:
            violations.append(f"future_scene_npc:{npc_name}")

    # PC duplication in scene_entities.
    pc_names_norm = {_normalize_name(name) for name in _current_pc_names(db) if name}
    scene_entities = result.get("scene_entities") or {}
    for entity_name in scene_entities.get("present_named_entities", []) or []:
        clean = _clean_text(entity_name)
        if clean and _normalize_name(clean) in pc_names_norm:
            violations.append(f"duplicated_pc_as_entity:{clean}")

    # Hidden clue title leaked without explicit clue_found.
    explicit_clue_found_norm = _normalize_name(_clean_text(updates.get("clue_found", "")))
    for clue_title in snap.get("hidden_clue_titles") or []:
        norm_title = _normalize_name(clue_title)
        if not norm_title or norm_title in current_scene_clues:
            continue
        if clue_title.lower() in narrative.lower() and norm_title != explicit_clue_found_norm:
            violations.append(f"undiscovered_clue_revealed:{clue_title}")

    # Later-act location mentions.
    # for act in blueprint.get("acts") or []:
    #     try:
    #         act_num = int(act.get("act", 0) or 0)
    #     except Exception:
    #         act_num = 0
    #     if act_num <= current_act_num:
    #         continue
    #     for scene in act.get("scenes") or []:
    #         loc = _clean_text(scene.get("location", ""))
    #         violations.extend(_actual_location_jump_violation(
    #             db,
    #             result,
    #             prefix="later_act_location",
    #         ))

    # Location update can be current or next reachable only.
    current_db_scene = db.get_current_story_scene()
    next_db_scene = db.get_next_story_scene(current_db_scene["id"]) if current_db_scene else None
    allowed_locations = {
        _normalize_name(snap.get("current_scene_location", "")),
        _normalize_name(_clean_text(current_scene_obj.get("location", ""))),
        _normalize_name(_clean_text((next_db_scene or {}).get("location_name", ""))),
    }
    allowed_locations = {x for x in allowed_locations if x}

    # new_location = _clean_text(updates.get("location_name", ""))
    # if new_location and _normalize_name(new_location) not in allowed_locations:
    #     violations.append(f"unearned_location_jump:{new_location}")

    # Prevent unsupported opening combat spikes.
    combat_action = result.get("combat_action") or {}
    current_scene_name = _clean_text(current_scene_obj.get("scene", "") or snap.get("current_scene", ""))
    current_scene_function = _clean_text(current_scene_obj.get("scene_function", "")).lower()
    current_scene_npcs = [_clean_text(x) for x in snap.get("current_scene_npc_names") or [] if _clean_text(x)]

    if combat_action.get("start_combat"):
        early_scene = _normalize_name(current_scene_name) in {"scene 1", "scene one"} or snap.get("current_act") == "1"
        no_scene_hostile_support = not current_scene_npcs
        if early_scene and no_scene_hostile_support and current_scene_function in {"investigation", "surface_inquiry"}:
            violations.append("premature_combat_start")

    violations.extend(_later_story_location_update_violation(
        db,
        result,
        prefix="later_act_location",
    ))
        
    return violations


def _later_story_location_update_violation(db: SessionDB, result: dict, *, prefix: str = "later_act_location") -> list[str]:
    """
    Only block a later scene location when the LLM actually sets it as state_updates.location_name.
    Mentioning future places in dialogue/narrative should not invalidate the turn.
    Unknown sublocations are allowed as emergent sublocations inside the current scene.
    """
    updates = result.get("state_updates") or {}
    new_location = _clean_text(updates.get("location_name", ""))
    if not new_location:
        return []

    current = db.get_current_story_scene()
    if not current:
        return []

    next_scene = db.get_next_story_scene(current["id"])
    allowed = {
        _normalize_name(current.get("location_name", "")),
        _normalize_name((next_scene or {}).get("location_name", "")),
    }
    allowed = {x for x in allowed if x}

    if _normalize_name(new_location) in allowed:
        return []

    try:
        current_act = int(current.get("act_no") or 1)
        current_scene_no = int(current.get("scene_no") or 1)
    except Exception:
        return []

    cur = db.conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT l.name, s.act_no, s.scene_no
        FROM story_scenes s
        LEFT JOIN locations l ON l.id = s.location_id
        WHERE l.name IS NOT NULL
          AND (
            s.act_no > ?
            OR (s.act_no = ? AND s.scene_no > ?)
          )
        """,
        (current_act, current_act, current_scene_no),
    ).fetchall()

    for row in rows:
        loc = _clean_text(row["name"])
        if loc and _same_locationish(loc, new_location):
            return [f"{prefix}:{loc}"]

    return []

def looks_like_valid_keeper_response(raw_text: str, parsed: dict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    required_keys = {"narrative", "suggested_actions", "roll_request"}
    return required_keys.issubset(parsed.keys())

def _actual_location_jump_violation(db: SessionDB, result: dict, *, prefix: str) -> list[str]:
    updates = result.get("state_updates") or {}
    new_location = _clean_text(updates.get("location_name", ""))

    if not new_location:
        return []

    current = db.get_current_story_scene()
    if not current:
        return []

    try:
        current_act = int(current.get("act_no") or 1)
        current_scene_no = int(current.get("scene_no") or 1)
    except Exception:
        return []

    cur = db.conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT l.name, s.act_no, s.scene_no
        FROM story_scenes s
        LEFT JOIN locations l ON l.id = s.location_id
        WHERE l.name IS NOT NULL
          AND (
            s.act_no > ?
            OR (s.act_no = ? AND s.scene_no > ?)
          )
        """,
        (current_act, current_act, current_scene_no),
    ).fetchall()

    for row in rows:
        loc = _clean_text(row["name"])
        if loc and _same_locationish(loc, new_location):
            return [f"{prefix}:{loc}"]

    return []


def validate_opening_scene_response(db: SessionDB, result: dict) -> list[str]:
    snap = _current_state_snapshot_for_validation(db)
    blueprint = snap["blueprint"] or {}
    current_scene_obj = snap["current_scene_obj"] or {}

    narrative = _clean_text(result.get("narrative") or "")
    updates = result.get("state_updates") or {}
    suggested_actions = result.get("suggested_actions") or []
    suggestions_blob = "\n".join(_clean_text(x) for x in suggested_actions if _clean_text(x))

    violations: list[str] = []
    if not narrative:
        violations.append("opening_missing_narrative")
    if not isinstance(suggested_actions, list) or len([x for x in suggested_actions if _clean_text(x)]) < 2:
        violations.append("opening_missing_actions")
    if _clean_text(updates.get("thread_progress", "")):
        violations.append("opening_thread_progress_too_early")

    start_location = _clean_text(current_scene_obj.get("location", "") or snap.get("current_scene_location", ""))
    new_location = _clean_text(updates.get("location_name", ""))
    if new_location and start_location and not _same_locationish(new_location, start_location):
        violations.append(f"opening_wrong_location:{new_location}")

    lowered = f"{narrative}\n{suggestions_blob}".lower()
    for forbidden, tag in (
        (_clean_text(blueprint.get("hidden_threat", "")), "opening_hidden_threat"),
        (_clean_text(blueprint.get("truth_the_players_never_suspect", "")), "opening_final_truth"),
    ):
        if forbidden and forbidden.lower() in lowered:
            violations.append(tag)

    try:
        current_act_num = int(snap["current_act"] or 1)
    except Exception:
        current_act_num = 1
    current_scene_no = int(snap.get("current_scene_no") or 0)

    for act in blueprint.get("acts") or []:
        try:
            act_num = int(act.get("act", 0) or 0)
        except Exception:
            act_num = 0
        for idx, scene in enumerate(act.get("scenes") or [], start=1):
            is_future = (act_num > current_act_num) or (act_num == current_act_num and idx > current_scene_no)
            if not is_future:
                continue

            loc = _clean_text(scene.get("location", ""))
            if loc and start_location and not _same_locationish(loc, start_location):
                violations.extend(_actual_location_jump_violation(
                    db,
                    result,
                    prefix="opening_later_act_location",
                ))
            for npc_name in scene.get("npc_present") or []:
                clean = _clean_text(npc_name)
                if clean and clean.lower() in lowered:
                    violations.append(f"opening_future_scene_npc:{clean}")

    return violations

def _player_facing_objective_hint(text: str) -> str:
    """
    Convert internal scene objective / dramatic question into safe player-facing hint.

    Avoid leaking GM/designer phrasing like:
    - What uncertainty, pressure, or decision defines this scene
    - Can the investigators...
    - What exactly were investigators officially sent to recover...
    """
    raw = _clean_text(text)
    if not raw:
        return ""

    lowered = raw.lower()

    # Do not expose designer/dramatic-question wording directly.
    banned_fragments = (
        "what uncertainty",
        "dramatic question",
        "defines this scene",
        "can the investigators",
        "what must be learned",
        "before moving on",
        "what exactly",
        "who already",
        "what changes by the end",
    )
    if any(x in lowered for x in banned_fragments):
        return ""

    # Questions are often internal dramatic questions. Do not show as action text.
    if raw.endswith("?"):
        return ""

    # Too abstract for player-facing UI.
    if len(raw) > 160:
        return ""

    return raw.strip(" .")


def build_opening_fallback_result(db: SessionDB) -> dict:
    cur = db.conn.cursor()
    current_objective = kv_get(cur, "current_objective", "")
    scene = db.get_current_story_scene()

    if scene:
        payload = scene.get("payload") or {}
        scene_name = _story_scene_display_name(scene) or "Opening Scene"
        location = _clean_text(scene.get("location_name", "")) or "the current scene"
        description = _clean_text(payload.get("description", "") or scene.get("location_description", ""))
        what_happens = _clean_text(payload.get("what_happens", ""))
        trigger = _clean_text(payload.get("trigger", ""))
        npcs = [
            _clean_text(n.get("name", ""))
            for n in db.list_story_scene_npcs(scene["id"])
            if _clean_text(n.get("name", ""))
        ]
        clues = [
            _clean_text(c.get("title", ""))
            for c in db.list_story_scene_clues(scene["id"], include_hidden=True)
            if _clean_text(c.get("title", ""))
        ]
    else:
        scene_name = "Opening Scene"
        location = "the current scene"
        description = ""
        what_happens = ""
        trigger = ""
        npcs = []
        clues = []

    # Keep fallback player-facing, not structural/designer-facing.
    narrative_parts = [f"You arrive at {location}."]

    if description:
        narrative_parts.append(description)

    # Prefer trigger/what_happens only if they read like fiction, not internal goal text.
    usable_trigger = _player_facing_objective_hint(trigger)
    usable_happens = _player_facing_objective_hint(what_happens)

    if usable_happens:
        narrative_parts.append(usable_happens)
    elif usable_trigger:
        narrative_parts.append(usable_trigger)
    elif scene_name:
        narrative_parts.append("The first signs of the case are already visible here.")

    objective_hint = _player_facing_objective_hint(_clean_opening_objective_text(current_objective))

    actions: list[str] = []

    if npcs:
        actions.append(f"Talk to {npcs[0]} about what seems wrong here")

    if clues:
        actions.append(f"Examine {clues[0]} closely")

    if objective_hint:
        actions.append(objective_hint)

    if location:
        actions.append(f"Survey {location} for anything immediately out of place")

    # Player-facing fallbacks only.
    for fallback in (
        "Review the briefing materials for the first concrete inconsistency",
        "Ask who has authority over this situation",
        "Look for the most obvious thing that does not fit the official explanation",
    ):
        if len(actions) >= 3:
            break
        actions.append(fallback)

    actions = _dedupe_keep_order(actions)[:3]

    return {
        "narrative": " ".join(x for x in narrative_parts if x),
        "suggested_actions": actions,
        "state_updates": {
            **_EMPTY_STATE_UPDATES,
            "location_name": location,
            "location_description": description,
        },
        "combat_action": dict(_EMPTY_COMBAT_ACTION),
        "roll_request": dict(_EMPTY_ROLL_REQUEST),
        "scene_entities": {"present_named_entities": npcs[:4]},
        "image_url": None,
        "generation_id": None,
    }

def _norm_fact(text: object) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"[^\w\s\-’']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
        "with", "from", "that", "this", "these", "those", "is", "are", "was",
        "were", "has", "have", "had", "investigators", "discovered", "found",
        "echoes", "invention",
    }
    ta = {t for t in re.findall(r"[a-zA-Z0-9’']{3,}", _norm_fact(a)) if t not in stop}
    tb = {t for t in re.findall(r"[a-zA-Z0-9’']{3,}", _norm_fact(b)) if t not in stop}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _recent_event_payload_text(db: SessionDB, event_types: set[str], limit: int = 24) -> list[str]:
    out: list[str] = []
    for e in db.list_events(limit=limit):
        if e.get("event_type") not in event_types:
            continue
        payload = e.get("payload") or {}
        blob = " ".join(str(v or "") for v in payload.values())
        if _clean_text(blob):
            out.append(_clean_text(blob))
    return out


def _is_duplicate_recent_progress(db: SessionDB, text: str, event_types: set[str], *, threshold: float = 0.72) -> bool:
    text = _clean_text(text)
    if not text:
        return False

    for prev in _recent_event_payload_text(db, event_types, limit=32):
        if _norm_fact(text) == _norm_fact(prev):
            return True
        if _token_overlap_ratio(text, prev) >= threshold:
            return True

    return False


def sanitize_llm_result_on_validation_failure(db: SessionDB, result: dict, violations: list[str]) -> dict:
    """
    Validation should protect state, not erase good narrative.

    Most validation failures are state/entity issues:
    - bad location update
    - hidden clue marked too early
    - PC duplicated as scene entity
    - future location mentioned in prose

    These should sanitize fields, not replace the whole response with stale DB fallback.
    """
    safe = dict(result or {})
    safe.setdefault("narrative", "")
    safe.setdefault("suggested_actions", [])
    safe.setdefault("state_updates", {})
    safe.setdefault("roll_request", dict(_EMPTY_ROLL_REQUEST))
    safe.setdefault("roll_resolution", None)
    safe.setdefault("combat_action", dict(_EMPTY_COMBAT_ACTION))
    safe.setdefault("scene_entities", {"present_named_entities": []})

    updates = {**_EMPTY_STATE_UPDATES, **dict(safe.get("state_updates") or {})}

    # Entity-only cleanup: never fallback narrative for this.
    if any(v.startswith(("duplicated_pc_as_entity:", "future_scene_npc:", "opening_future_scene_npc:")) for v in violations):
        pc_names = {_normalize_name(name) for name in _current_pc_names(db) if name}
        entities = safe.get("scene_entities") or {}
        names = entities.get("present_named_entities") or []
        filtered = [
            _clean_text(name)
            for name in names
            if _clean_text(name) and _normalize_name(_clean_text(name)) not in pc_names
        ]
        safe["scene_entities"] = {"present_named_entities": filtered}

    # Location-state cleanup only.
    if any(v.startswith((
        "unearned_location_jump:",
        "opening_wrong_location:",
        "later_act_location:",
        "opening_later_act_location:",
    )) for v in violations):
        updates["location_name"] = ""
        updates["location_description"] = ""

    # Clue-state cleanup only.
    if any(v.startswith("undiscovered_clue_revealed:") for v in violations):
        updates["clue_found"] = ""
        updates["clue_content"] = ""

    # Premature combat cleanup only.
    if any(v == "premature_combat_start" for v in violations):
        safe["combat_action"] = dict(_EMPTY_COMBAT_ACTION)

    safe["state_updates"] = updates

    # Only use fallback for truly unusable model output.
    narrative = _clean_text(safe.get("narrative", ""))
    actions = safe.get("suggested_actions") or []
    if not narrative or len([a for a in actions if _clean_text(a)]) == 0:
        logger.warning("Sanitizer fallback used only because response had no usable narrative/actions.")
        return build_opening_fallback_result(db)

    return safe


def _text_match_loose(a: str, b: str) -> bool:
    na = _normalize_name(a or "")
    nb = _normalize_name(b or "")
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na

def advance_story_progress(db: SessionDB, result: dict) -> None:
    current_scene = db.get_current_story_scene()
    if not current_scene:
        return

    next_scene = db.get_next_story_scene(current_scene["id"])
    if not next_scene:
        return

    # Avoid double-advancing if something already changed the pointer.
    movement_from_scene_id = _clean_text(result.get("_movement_from_story_scene_id", ""))
    if movement_from_scene_id and movement_from_scene_id != _clean_text(current_scene.get("id", "")):
        logger.info(
            "STORY_ADVANCE_SKIPPED_STALE_FLAG from_scene=%r current_scene=%r",
            movement_from_scene_id,
            current_scene.get("id", ""),
        )
        return

    updates = result.get("state_updates") or {}
    new_location = _clean_text(updates.get("location_name", ""))

    movement_intent = bool(result.get("_player_movement_intent"))
    explicit_advance = bool(result.get("_scene_advance_allowed"))

    if not (movement_intent or explicit_advance):
        return

    next_location = _clean_text(next_scene.get("location_name", ""))

    if not explicit_advance:
        if not new_location or not next_location or not _same_locationish(new_location, next_location):
            logger.info(
                "STORY_ADVANCE_BLOCKED location_mismatch movement=%s new_location=%r next_location=%r current_scene=%r",
                movement_intent,
                new_location,
                next_location,
                current_scene.get("name", ""),
            )
            return

    logger.info(
        "STORY_SCENE_ADVANCED act=%s scene=%s location=%r -> act=%s scene=%s location=%r",
        current_scene.get("act_no"),
        current_scene.get("name"),
        current_scene.get("location_name", ""),
        next_scene.get("act_no"),
        next_scene.get("name"),
        next_scene.get("location_name", ""),
    )

    db.mark_story_scene_resolved(current_scene["id"])
    db.set_current_story_scene(next_scene["id"])

    # set_current_story_scene() already mirrors these in your current db_session.py,
    # but keep this as a safe compatibility mirror.
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        ("current_act", str(next_scene["act_no"])),
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        ("current_scene", str(next_scene["name"])),
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        ("current_scene_location", _clean_text(next_scene.get("location_name", ""))),
    )

    primary_objective = db.get_story_scene_primary_objective(next_scene["id"])
    if primary_objective:
        db.conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            ("current_objective", primary_objective),
        )

    db.conn.commit()


def _find_actor_for_state_update(db: SessionDB, target_name: str) -> dict | None:
    actors = db.list_actors("PC") + db.list_actors("NPC") + db.list_actors("ENEMY")
    wanted = _clean_text(target_name).lower()
    if wanted:
        exact = next((a for a in actors if a["name"].strip().lower() == wanted), None)
        if exact:
            return exact
        fuzzy = next((a for a in actors if wanted in a["name"].strip().lower()), None)
        if fuzzy:
            return fuzzy

    pcs = db.list_actors("PC")
    return pcs[0] if pcs else None


def _apply_actor_changes(db: SessionDB, result: dict, updates: dict) -> None:
    target_name = _clean_text(updates.get("character_name", ""))
    has_stats = any(int(updates.get(k, 0) or 0) != 0 for k in ("hp_change", "sanity_change", "mp_change"))
    has_inventory = bool(_clean_text(updates.get("inventory_add", "")) or _clean_text(updates.get("inventory_remove", "")))

    if not (target_name or has_stats or has_inventory):
        return

    target = _find_actor_for_state_update(db, target_name)
    if not target:
        return

    hp_change = int(updates.get("hp_change", 0) or 0)
    san_change = int(updates.get("sanity_change", 0) or 0)
    mp_change = int(updates.get("mp_change", 0) or 0)

    if has_stats:
        cur_hp = target.get("hp")
        cur_san = target.get("san")
        cur_mp = target.get("mp")

        new_hp = max(0, int(cur_hp) + hp_change) if cur_hp is not None else None
        new_san = max(0, int(cur_san) + san_change) if cur_san is not None else None
        new_mp = max(0, int(cur_mp) + mp_change) if cur_mp is not None else None

        status = target.get("status", "ok")
        if cur_hp is not None and hp_change != 0 and new_hp == 0:
            status = "dead"
        elif cur_san is not None and san_change != 0 and new_san == 0:
            status = "insane"
        elif cur_hp is not None and hp_change < 0:
            status = "injured"

        patch_kwargs = {"actor_id": target["id"], "status": status}
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

    item_add = _clean_text(updates.get("inventory_add", ""))
    if item_add:
        db.add_actor_item(target["id"], item_add)
        db.log_event("INVENTORY", {"actor": target["name"], "added": item_add})

    item_remove = _clean_text(updates.get("inventory_remove", ""))
    if item_remove:
        db.remove_actor_item(target["id"], item_remove)
        db.log_event("INVENTORY", {"actor": target["name"], "removed": item_remove})


def _current_pc_location_names(db: SessionDB) -> set[str]:
    out: set[str] = set()
    cur = db.conn.cursor()
    for pc in db.list_actors("PC"):
        loc_id = pc.get("location_id")
        if not loc_id:
            continue
        row = cur.execute("SELECT name FROM locations WHERE id=?", (loc_id,)).fetchone()
        if row and _clean_text(row["name"]):
            out.add(_clean_text(row["name"]))
    return out


def _apply_location_change(db: SessionDB, updates: dict) -> None:
    new_location = _clean_text(updates.get("location_name", ""))
    if not new_location:
        return

    current_locations = _current_pc_location_names(db)
    if any(_same_locationish(new_location, loc) for loc in current_locations):
        db.log_event("DUPLICATE_LOCATION_SUPPRESSED", {
            "location": new_location,
            "current_locations": list(current_locations),
        })
        return

    if _is_duplicate_recent_progress(db, new_location, {"LOCATION_CHANGE"}, threshold=0.82):
        db.log_event("DUPLICATE_LOCATION_SUPPRESSED", {"location": new_location})
        return

    existing = db.get_location_by_name(new_location)
    loc_id = existing["id"] if existing else db.upsert_location(
        name=new_location,
        description=_clean_text(updates.get("location_description", "")),
    )
    for pc in db.list_actors("PC"):
        db.patch_actor(actor_id=pc["id"], location_id=loc_id)

    db.log_event("LOCATION_CHANGE", {"location": new_location})

def _apply_clue_discovery(db: SessionDB, updates: dict) -> None:
    current_scene = db.get_current_story_scene()
    current_scene_id = current_scene["id"] if current_scene else None

    clue_found = _clean_text(updates.get("clue_found", ""))
    clue_content = _clean_text(updates.get("clue_content", ""))
    if not clue_found:
        return

    combined = f"{clue_found}: {clue_content}"

    # Do not let repeated rediscovery reset anti-stall logic.
    if _is_duplicate_recent_progress(db, combined, {"CLUE_FOUND"}, threshold=0.70):
        db.log_event("DUPLICATE_CLUE_SUPPRESSED", {
            "clue": clue_found,
            "content": clue_content[:300],
        })
        return

    # Also check already found clues by title/content.
    for existing in db.list_clues(status="found", limit=80):
        existing_title = _clean_text(existing.get("title", ""))
        existing_content = _clean_text(existing.get("content", ""))

        same_title = _normalize_name(existing_title) == _normalize_name(clue_found)
        similar_content = _token_overlap_ratio(clue_content, existing_content) >= 0.72

        if same_title and similar_content:
            db.log_event("DUPLICATE_CLUE_SUPPRESSED", {
                "clue": clue_found,
                "matched_existing": existing_title,
            })
            return

    scene_clues = db.list_story_scene_clues(current_scene_id, include_hidden=True) if current_scene_id else []
    match = next((c for c in scene_clues if _text_match_loose(clue_found, c["title"])), None)

    if match:
        canonical_title = _clean_text(match.get("title", "")) or clue_found
        canonical_content = clue_content or _clean_text(match.get("content", ""))
        db.upsert_clue(
            clue_id=match["id"],
            title=canonical_title,
            content=canonical_content,
            status="found",
            location_id=match.get("location_id"),
        )
        updates["clue_found"] = canonical_title
        updates["clue_content"] = canonical_content
        db.log_event("CLUE_FOUND", {"clue": canonical_title, "content": canonical_content[:500]})
        return

    location_id = current_scene.get("location_id") if current_scene else None
    cid = db.upsert_clue(
        title=clue_found,
        content=clue_content,
        status="found",
        location_id=location_id,
    )
    if current_scene_id:
        db.link_story_scene_clue(current_scene_id, cid)

    db.log_event("CLUE_FOUND", {"clue": clue_found, "content": clue_content[:500]})


def _apply_thread_progress(db: SessionDB, updates: dict) -> None:
    thread_note = _clean_text(updates.get("thread_progress", ""))
    if not thread_note:
        return

    if _is_duplicate_recent_progress(db, thread_note, {"THREAD_PROGRESS"}, threshold=0.70):
        db.log_event("DUPLICATE_THREAD_PROGRESS_SUPPRESSED", {
            "note": thread_note[:500],
        })
        return

    current_scene = db.get_current_story_scene()
    current_scene_id = current_scene["id"] if current_scene else None
    scene_threads = db.list_story_scene_threads(current_scene_id) if current_scene_id else []
    target_thread = scene_threads[0] if scene_threads else None

    if target_thread:
        new_progress = min(int(target_thread["progress"]) + 1, int(target_thread["max_progress"]))
        db.upsert_thread(
            thread_id=target_thread["id"],
            name=target_thread["name"],
            progress=new_progress,
            max_progress=int(target_thread["max_progress"]),
            stakes=target_thread.get("stakes", ""),
        )
        db.log_event("THREAD_PROGRESS", {
            "thread": target_thread["name"],
            "progress": new_progress,
            "note": thread_note,
        })
    else:
        db.log_event("THREAD_PROGRESS", {
            "thread": thread_note,
            "progress": None,
            "note": thread_note,
        })


def apply_state_updates(db: SessionDB, result: dict) -> None:
    """
    Apply canonical English state_updates to SQLite.

    Display translation must happen after this. Do not feed translated display
    strings back into this function.
    """
    updates = {**_EMPTY_STATE_UPDATES, **dict(result.get("state_updates") or {})}
    result["state_updates"] = updates

    _apply_actor_changes(db, result, updates)
    _apply_location_change(db, updates)
    _apply_clue_discovery(db, updates)
    _apply_thread_progress(db, updates)

    current_scene = db.get_current_story_scene()
    if current_scene:
        cur = db.conn.cursor()
        current_objective = kv_get(cur, "current_objective", "").strip()
        if not current_objective:
            scene_objective = db.get_story_scene_primary_objective(current_scene["id"])
            if scene_objective:
                kv_set(db, "current_objective", scene_objective)

    advance_story_progress(db, result)
