import logging
import re
#pylint: disable=import-error
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


def _clean_opening_objective_text(text: str) -> str:
    text = str(text or "").strip()

    # remove explicit act labels
    text = re.sub(r"^\s*Act\s+\d+\s*:\s*", "", text, flags=re.IGNORECASE)

    # if objective was generated in old machine format, keep only the actual lead
    m = re.search(r"follow this lead:\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .")

    m = re.search(r"establish what is happening here\s*[—\-]\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .")

    return text.strip(" .")


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


def compact_blueprint_text(
    blueprint: dict | None,
    *,
    current_act: str = "1",
    current_scene: str = "",
    met_npc_names: list[str] | None = None,
    found_clue_titles: list[str] | None = None,
    include_next_hint: bool = True,
) -> str:
    """
    Return only the playable slice of the scenario blueprint.

    Design goals:
    - acts.scenes remain the primary narrative skeleton
    - registries are compact canonical references
    - only scene-relevant registry items are surfaced
    - do NOT expose hidden_threat, core_mystery, truth_the_players_never_suspect,
      later-act content, unseen NPC secrets, or unseen hidden clues
    """
    if not isinstance(blueprint, dict) or not blueprint:
        return "(none)"

    def _norm(x: str) -> str:
        return _normalize_name(str(x or ""))

    def _match_registry_items_by_names(items: list[dict], names: list[str], key: str = "name") -> list[dict]:
        wanted = {_norm(x) for x in (names or []) if str(x).strip()}
        if not wanted:
            return []
        out = []
        for item in items or []:
            value = _norm(item.get(key, ""))
            if value and value in wanted:
                out.append(item)
        return out

    met_npc_names_norm = {_norm(x) for x in (met_npc_names or []) if x}
    found_clue_titles_norm = {_norm(x) for x in (found_clue_titles or []) if x}

    lines: list[str] = []

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
            "scene",
            "location",
            "scene_function",
            "dramatic_question",
            "entry_condition",
            "exit_condition",
            "trigger",
            "description",
            "what_happens",
            "pressure_if_delayed",
            "threat_level",
        ]:
            value = current_scene_obj.get(field)
            if value:
                lines.append(f"- {field}={value}")

        visible_scene_npcs = [str(x) for x in (current_scene_obj.get("npc_present") or []) if x]
        visible_scene_clues = [str(x) for x in (current_scene_obj.get("clues_available") or []) if x]

        if visible_scene_npcs:
            lines.append("- scene_npcs=" + ", ".join(visible_scene_npcs))
        if visible_scene_clues:
            lines.append("- scene_clues=" + ", ".join(visible_scene_clues))
    else:
        visible_scene_npcs = []
        visible_scene_clues = []

    if include_next_hint and next_scene_obj:
        lines.append("next_reachable_scene_hint:")
        for field in ["scene", "location", "trigger", "threat_level"]:
            value = next_scene_obj.get(field)
            if value:
                lines.append(f"- {field}={value}")

    current_location = str(current_scene_obj.get("location", "") or "").strip() if current_scene_obj else ""

    current_scene_npc_norm = {_norm(x) for x in visible_scene_npcs}
    allowed_npc_names = met_npc_names_norm | current_scene_npc_norm

    current_scene_clue_norm = {_norm(x) for x in visible_scene_clues}
    allowed_clue_titles = found_clue_titles_norm | current_scene_clue_norm

    locations = blueprint.get("locations") or []
    npcs = blueprint.get("npcs") or []
    clues = blueprint.get("clues") or []
    plot_threads = blueprint.get("plot_threads") or []

    matched_locations = _match_registry_items_by_names(
        locations,
        [current_location] if current_location else [],
        key="name",
    )
    if matched_locations:
        lines.append("current_location_registry:")
        for loc in matched_locations:
            parts = [f"name={loc.get('name', '')}"]
            if loc.get("tags"):
                parts.append(f"tags={loc.get('tags', '')}")
            if loc.get("hidden"):
                parts.append(f"hidden={loc.get('hidden', '')}")
            lines.append("- " + "; ".join(parts))

    if allowed_npc_names and npcs:
        lines.append("current_npc_registry:")
        for npc in npcs:
            npc_name = str(npc.get("name", "") or "")
            if _norm(npc_name) not in allowed_npc_names:
                continue

            parts = [f"name={npc_name}"]
            if npc.get("role"):
                parts.append(f"role={npc['role']}")
            if npc.get("secret"):
                parts.append(f"secret={npc['secret']}")
            if npc.get("motivation"):
                parts.append(f"motivation={npc['motivation']}")
            lines.append("- " + "; ".join(parts))

    if allowed_clue_titles and clues:
        lines.append("current_clue_registry:")
        for clue in clues:
            clue_title = str(clue.get("title", "") or "")
            if _norm(clue_title) not in allowed_clue_titles:
                continue

            parts = [f"title={clue_title}"]
            if clue.get("content"):
                parts.append(f"content={clue['content']}")
            if clue.get("true_meaning"):
                parts.append(f"true_meaning={clue['true_meaning']}")
            if clue.get("location"):
                parts.append(f"location={clue['location']}")
            lines.append("- " + "; ".join(parts))

    if plot_threads:
        lines.append("active_plot_threads:")
        for th in plot_threads[:2]:
            parts = []
            if th.get("name"):
                parts.append(f"name={th['name']}")
            if th.get("stakes"):
                parts.append(f"stakes={th['stakes']}")
            if th.get("steps") not in (None, ""):
                parts.append(f"steps={th['steps']}")
            if parts:
                lines.append("- " + "; ".join(parts))

    return "\n".join(lines) if lines else "(none)"


def build_authoritative_context(
    db: SessionDB,
    *,
    campaign_atoms: str = "",
    themes: str = "STANDARD",
    include_next_hint: bool = True,
) -> tuple[str, str]:
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
            f"{compact_blueprint_text(blueprint, current_act=current_act, current_scene=current_scene, met_npc_names=met_npc_names, found_clue_titles=found_clue_titles, include_next_hint=include_next_hint)}\n"
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


def _mentions_location_explicitly(location: str, *texts: str) -> bool:
    loc_norm = _normalize_name(location)
    if not loc_norm:
        return False

    for text in texts:
        text_norm = _normalize_name(text or "")
        if loc_norm and loc_norm in text_norm:
            return True
    return False

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

    start_location_raw = str(current_scene_obj.get("location", "") or snap["current_scene_location"] or "")
    new_location_raw = str(updates.get("location_name", "") or "")

    if new_location_raw and start_location_raw and not _same_locationish(new_location_raw, start_location_raw):
        violations.append(f"opening_wrong_location:{updates.get('location_name', '')}")

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

    try:
        current_act_num = int(snap["current_act"] or 1)
    except Exception:
        current_act_num = 1

    start_location_norm = str(current_scene_obj.get("location", "") or snap["current_scene_location"] or "")

    for act in blueprint.get("acts") or []:
        try:
            act_num = int(act.get("act", 0) or 0)
        except Exception:
            act_num = 0
        if act_num <= current_act_num:
            continue

        for scene in act.get("scenes") or []:
            loc = str(scene.get("location", "") or "").strip()
            if not loc:
                continue
            if _same_locationish(loc, start_location_norm):
                continue
            if _mentions_location_explicitly(loc, narrative, suggestions_blob):
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
    clean_objective = _clean_opening_objective_text(current_objective)
    if clean_objective:
        narrative_parts.append(f"One immediate lead stands out: {clean_objective}.")

    suggested_actions = []

    if npcs:
        suggested_actions.append(f"Talk to {npcs[0]} about what seems wrong here")

    if clues:
        suggested_actions.append(f"Examine {clues[0]} closely")

    if clean_objective:
        suggested_actions.append(clean_objective)

    if location:
        suggested_actions.append(f"Survey {location} for anything immediately out of place")

    if description:
        suggested_actions.append("Focus on the most suspicious visible detail in the room")

    if what_happens:
        suggested_actions.append("Test the anomaly that is already unfolding here")

    # dedupe while preserving order
    deduped = []
    seen = set()
    for item in suggested_actions:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())

    suggested_actions = deduped[:3]

    while len(suggested_actions) < 3:
        fallback_pool = [
            "Question the nearest person about what just happened",
            "Inspect the most abnormal machine, object, or trace",
            "Compare what you see with what should normally be here",
        ]
        for item in fallback_pool:
            if len(suggested_actions) >= 3:
                break
            if item.lower() not in {x.lower() for x in suggested_actions}:
                suggested_actions.append(item)

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


def _text_match_loose(a: str, b: str) -> bool:
    na = _normalize_name(a or "")
    nb = _normalize_name(b or "")
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _derive_scene_objective(scene: dict) -> str:
    for key in ("trigger", "dramatic_question", "what_happens", "exit_condition"):
        value = str(scene.get(key, "") or "").strip().rstrip(". ?")
        if value:
            return value
    return ""


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
    thread_name = str(updates.get("thread_progress", "") or "").strip()

    scene_clues = [str(x) for x in (current_scene_obj.get("clues_available") or []) if x]
    scene_location = str(current_scene_obj.get("location", "") or "").strip()

    payoff_hit = False

    if clue_found and any(_text_match_loose(clue_found, c) for c in scene_clues):
        payoff_hit = True

    if thread_name:
        payoff_hit = True

    if location_name and not _same_locationish(location_name, scene_location):
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

            if scene_idx + 1 < len(scenes):
                next_scene = scenes[scene_idx + 1]
                kv_set(db, "current_scene", str(next_scene.get("scene", "") or ""))
                kv_set(db, "current_scene_location", str(next_scene.get("location", "") or ""))
                kv_set(db, "current_objective", _derive_scene_objective(next_scene))
                return

            if act_idx + 1 < len(acts):
                next_act = acts[act_idx + 1]
                next_scenes = next_act.get("scenes") or []
                kv_set(db, "current_act", str(next_act.get("act", current_act_num + 1)))
                if next_scenes:
                    next_scene = next_scenes[0]
                    kv_set(db, "current_scene", str(next_scene.get("scene", "") or ""))
                    kv_set(db, "current_scene_location", str(next_scene.get("location", "") or ""))
                    kv_set(db, "current_objective", _derive_scene_objective(next_scene))
                return

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
