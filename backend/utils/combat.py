import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple
#pylint: disable=import-error
from utils.db_session import SessionDB

logger = logging.getLogger("keeper_ai.combat")


SUCCESS_RANK = {
    "fumble": -1,
    "failure": 0,
    "regular_success": 1,
    "hard_success": 2,
    "extreme_success": 3,
    "critical_success": 4,
}

DEFAULT_HUMAN_FIGHTING = 25
DEFAULT_ENEMY_FIGHTING = 40


# ─────────────────────────────────────────────
# kv helpers
# ─────────────────────────────────────────────

def _kv_get(cur, key: str, default: str = "") -> str:
    cur.execute("SELECT value FROM kv_store WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


def _kv_set(db: SessionDB, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        (key, value),
    )
    db.conn.commit()


# ─────────────────────────────────────────────
# generic helpers
# ─────────────────────────────────────────────

def is_combat_trigger(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    triggers = (
        "attack", "attacker", "fight", "shoot", "fire", "stab", "slash", "cut",
        "kill", "hit", "strike", "grab", "disarm", "tackle", "punch", "kick",
        "bite", "claw", "swing", "open fire", "aim and shoot"
    )
    return any(t in text for t in triggers)


def _safe_json_loads(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _percentile_success(roll: int, value: int) -> str:
    value = max(0, int(value))
    if roll == 100 or (value < 50 and roll >= 96):
        return "fumble"
    if roll == 1:
        return "critical_success"
    if roll <= max(1, value // 5):
        return "extreme_success"
    if roll <= max(1, value // 2):
        return "hard_success"
    if roll <= value:
        return "regular_success"
    return "failure"


def _success_rank(outcome: str) -> int:
    return SUCCESS_RANK.get((outcome or "").strip().lower(), 0)


def _roll_percentile(*, bonus_dice: int = 0, penalty_dice: int = 0) -> int:
    if bonus_dice > 0 and penalty_dice > 0:
        cancel = min(bonus_dice, penalty_dice)
        bonus_dice -= cancel
        penalty_dice -= cancel

    units = random.randint(0, 9)
    tens_count = 1 + max(bonus_dice, penalty_dice)
    tens_pool = [random.randint(0, 9) for _ in range(tens_count)]

    if bonus_dice > 0:
        tens = min(tens_pool)
    elif penalty_dice > 0:
        tens = max(tens_pool)
    else:
        tens = tens_pool[0]

    result = tens * 10 + units
    return 100 if result == 0 else result


def _parse_dice(expr: str) -> Tuple[int, int, int]:
    m = re.fullmatch(r"\s*(\d+)[dD](\d+)(?:\+(\d+))?\s*", expr or "")
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def _roll_dice(expr: str) -> int:
    n, d, bonus = _parse_dice(expr)
    if n <= 0 or d <= 0:
        return 0
    return sum(random.randint(1, d) for _ in range(n)) + bonus


def _max_dice(expr: str) -> int:
    n, d, bonus = _parse_dice(expr)
    if n <= 0 or d <= 0:
        return 0
    return n * d + bonus


# ─────────────────────────────────────────────
# actor helpers
# ─────────────────────────────────────────────

def _find_actor_by_name(db: SessionDB, name: str) -> Optional[Dict[str, Any]]:
    target = (name or "").strip().lower()
    if not target:
        return None

    actors = db.list_actors()
    exact = next((a for a in actors if a["name"].strip().lower() == target), None)
    if exact:
        return exact
    return next((a for a in actors if target in a["name"].strip().lower()), None)


def _actor_skill(actor: Dict[str, Any], skill_name: str) -> int:
    target = (skill_name or "").strip().lower()
    if not target:
        return 0
    skills = actor.get("skills") or {}
    for name, value in skills.items():
        if name.strip().lower() == target:
            return int(value)
    return 0


def _actor_stat(actor: Dict[str, Any], name: str, default: int = 0) -> int:
    value = actor.get(name)
    return default if value is None else int(value)


def _calc_max_hp(actor: Dict[str, Any]) -> int:
    con = actor.get("con")
    siz = actor.get("siz")
    if con is not None and siz is not None:
        return max(1, round((int(con) + int(siz)) / 10))
    hp = actor.get("hp")
    if hp is not None:
        return max(1, int(hp))
    return 10


def _calc_build(actor: Dict[str, Any]) -> int:
    total = _actor_stat(actor, "str") + _actor_stat(actor, "siz")
    if total <= 64:
        return -2
    if total <= 84:
        return -1
    if total <= 124:
        return 0
    if total <= 164:
        return 1
    if total <= 204:
        return 2
    return 3 + max(0, (total - 205) // 80)


def _calc_damage_bonus_expr(actor: Dict[str, Any]) -> str:
    build = _calc_build(actor)
    if build <= -2:
        return "-2"
    if build == -1:
        return "-1"
    if build == 0:
        return "0"
    if build == 1:
        return "1D4"
    return "1D6"


def _combined_damage_expr(base_expr: str, bonus_expr: str) -> str:
    base_expr = (base_expr or "").strip()
    bonus_expr = (bonus_expr or "0").strip()

    if not base_expr:
        return bonus_expr if bonus_expr not in ("0", "", "-1", "-2") else "0"

    if bonus_expr in ("0", "", "-1", "-2"):
        return base_expr

    return f"{base_expr}+{bonus_expr}"


def _default_weapon_damage(action_type: str, weapon_name: str) -> str:
    w = (weapon_name or "").strip().lower()
    if action_type == "attack_firearm":
        if "shotgun" in w:
            return "4D6"
        if "rifle" in w:
            return "2D6+4"
        return "1D10"

    if "knife" in w:
        return "1D4"
    if "machete" in w:
        return "1D8"
    if "bat" in w:
        return "1D8"
    if "club" in w:
        return "1D6"
    return "1D3"


def ensure_actor_combat_ready(db: SessionDB, actor_id: int) -> None:
    actor = next((a for a in db.list_actors() if a["id"] == actor_id), None)
    if not actor:
        return

    stats_patch: Dict[str, int] = {}
    for stat_name, fallback in (
        ("str", 50), ("con", 50), ("dex", 50), ("int", 50),
        ("pow", 50), ("app", 40), ("siz", 50), ("edu", 40),
    ):
        if actor.get(stat_name) is None:
            stats_patch[stat_name] = fallback

    if stats_patch:
        db.patch_actor(actor_id=actor_id, stats=stats_patch)
        actor = next((a for a in db.list_actors() if a["id"] == actor_id), actor)

    if actor.get("hp") is None:
        hp = _calc_max_hp({**actor, **stats_patch})
        db.patch_actor(actor_id=actor_id, hp=hp)
        actor = next((a for a in db.list_actors() if a["id"] == actor_id), actor)

    skills = actor.get("skills") or {}
    if "Dodge" not in skills:
        db.set_skill(actor_id, "Dodge", max(20, _actor_stat(actor, "dex") // 2))

    if "Fighting (Brawl)" not in skills:
        default = DEFAULT_ENEMY_FIGHTING if actor.get("kind") == "ENEMY" else DEFAULT_HUMAN_FIGHTING
        db.set_skill(actor_id, "Fighting (Brawl)", default)


def ensure_scene_combatants_ready(db: SessionDB) -> None:
    for actor in db.list_actors():
        if actor.get("kind") in ("PC", "NPC", "ENEMY"):
            ensure_actor_combat_ready(db, actor["id"])


# ─────────────────────────────────────────────
# combat state
# ─────────────────────────────────────────────

def get_combat_state(db: SessionDB) -> Dict[str, Any]:
    cur = db.conn.cursor()
    raw = _kv_get(cur, "combat_state_json", "")
    return _safe_json_loads(raw, {
        "active": False,
        "round": 0,
        "phase": "idle",
        "participants": [],
        "turn_index": 0,
        "current_actor_id": None,
        "pending_attack": None,
        "log": [],
    })


def set_combat_state(db: SessionDB, state: Dict[str, Any]) -> None:
    _kv_set(db, "combat_state_json", json.dumps(state, ensure_ascii=False))


def build_initiative_order(db: SessionDB, actor_ids: List[int], *, readied_firearm_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    readied_firearm_ids = set(readied_firearm_ids or [])
    actors = {a["id"]: a for a in db.list_actors()}
    participants = []

    for actor_id in actor_ids:
        actor = actors.get(actor_id)
        if not actor:
            continue
        dex = _actor_stat(actor, "dex", 50)
        initiative = dex + 50 if actor_id in readied_firearm_ids else dex
        participants.append({
            "actor_id": actor_id,
            "name": actor["name"],
            "kind": actor.get("kind", ""),
            "dex": dex,
            "initiative": initiative,
            "has_acted": False,
            "responses_this_round": 0,
            "forfeit_this_action": False,
            "forfeit_next_action": False,
            "readied_firearm": actor_id in readied_firearm_ids,
        })

    participants.sort(key=lambda p: (-p["initiative"], -p["dex"], p["name"].lower()))
    return participants


def _living_actor_ids(db: SessionDB) -> List[int]:
    ids = []
    for actor in db.list_actors():
        if actor.get("kind") not in ("PC", "NPC", "ENEMY"):
            continue
        if actor.get("status") in ("dead", "dying"):
            continue
        ids.append(actor["id"])
    return ids


def _visible_hostile_ids(db: SessionDB) -> List[int]:
    return [a["id"] for a in db.list_actors("ENEMY") if a.get("status") not in ("dead", "dying")]


def maybe_start_combat(db: SessionDB, llm_result: Dict[str, Any]) -> Dict[str, Any]:
    state = get_combat_state(db)
    if state.get("active"):
        return state

    combat_action = llm_result.get("combat_action") or {}
    if not combat_action.get("start_combat"):
        return state

    ensure_scene_combatants_ready(db)

    actor_name = combat_action.get("actor_name", "")
    target_name = combat_action.get("target_name", "")

    actor = _find_actor_by_name(db, actor_name) if actor_name else None
    target = _find_actor_by_name(db, target_name) if target_name else None

    actor_ids: List[int] = []
    if actor:
        actor_ids.append(actor["id"])
    if target:
        actor_ids.append(target["id"])

    hostile_ids = _visible_hostile_ids(db)
    for hid in hostile_ids:
        if hid not in actor_ids:
            actor_ids.append(hid)

    for pc in db.list_actors("PC"):
        if pc.get("status") not in ("dead", "dying") and pc["id"] not in actor_ids:
            actor_ids.append(pc["id"])

    if len(actor_ids) < 2:
        return state

    participants = build_initiative_order(db, actor_ids)
    new_state = {
        "active": True,
        "round": 1,
        "phase": "await_action",
        "participants": participants,
        "turn_index": 0,
        "current_actor_id": participants[0]["actor_id"] if participants else None,
        "pending_attack": None,
        "log": [{
            "type": "combat_started",
            "round": 1,
        }],
    }
    set_combat_state(db, new_state)
    db.log_event("COMBAT_START", {
        "participants": [p["name"] for p in participants]
    })
    return new_state


def get_current_combat_actor(db: SessionDB) -> Optional[Dict[str, Any]]:
    state = get_combat_state(db)
    if not state.get("active"):
        return None
    actor_id = state.get("current_actor_id")
    if actor_id is None:
        return None
    return next((a for a in db.list_actors() if a["id"] == actor_id), None)


def _participant_index(state: Dict[str, Any], actor_id: int) -> int:
    for i, p in enumerate(state.get("participants", [])):
        if p["actor_id"] == actor_id:
            return i
    return -1


def _alive_participant_ids(db: SessionDB, state: Dict[str, Any]) -> List[int]:
    actors = {a["id"]: a for a in db.list_actors()}
    ids = []
    for p in state.get("participants", []):
        actor = actors.get(p["actor_id"])
        if not actor:
            continue
        if actor.get("status") in ("dead", "dying"):
            continue
        ids.append(actor["id"])
    return ids


def advance_turn(db: SessionDB) -> Dict[str, Any]:
    state = get_combat_state(db)
    if not state.get("active"):
        return state

    participants = state.get("participants", [])
    if not participants:
        state["active"] = False
        state["phase"] = "idle"
        set_combat_state(db, state)
        return state

    actors = {a["id"]: a for a in db.list_actors()}
    alive_ids = set(_alive_participant_ids(db, state))

    start = state.get("turn_index", 0)
    for offset in range(1, len(participants) + 1):
        idx = (start + offset) % len(participants)
        p = participants[idx]
        actor = actors.get(p["actor_id"])
        if not actor or p["actor_id"] not in alive_ids:
            continue
        if p.get("has_acted"):
            continue

        state["turn_index"] = idx
        state["current_actor_id"] = p["actor_id"]
        state["phase"] = "await_action"
        set_combat_state(db, state)
        return state

    return advance_round(db)


def advance_round(db: SessionDB) -> Dict[str, Any]:
    state = get_combat_state(db)
    if not state.get("active"):
        return state

    actors = {a["id"]: a for a in db.list_actors()}
    for p in state.get("participants", []):
        p["has_acted"] = False
        p["responses_this_round"] = 0
        p["forfeit_this_action"] = bool(p.get("forfeit_next_action", False))
        p["forfeit_next_action"] = False
        actor = actors.get(p["actor_id"])
        if not actor or actor.get("status") in ("dead", "dying"):
            p["has_acted"] = True

    state["round"] = int(state.get("round", 1)) + 1

    for idx, p in enumerate(state.get("participants", [])):
        if not p.get("has_acted"):
            state["turn_index"] = idx
            state["current_actor_id"] = p["actor_id"]
            state["phase"] = "await_action"
            state.setdefault("log", []).append({
                "type": "new_round",
                "round": state["round"],
            })
            set_combat_state(db, state)
            return state

    state["active"] = False
    state["phase"] = "idle"
    set_combat_state(db, state)
    return state


def end_combat(db: SessionDB, reason: str = "") -> None:
    state = get_combat_state(db)
    if not state.get("active"):
        return
    state["active"] = False
    state["phase"] = "idle"
    state["pending_attack"] = None
    state.setdefault("log", []).append({
        "type": "combat_ended",
        "round": state.get("round", 0),
        "reason": reason,
    })
    set_combat_state(db, state)
    db.log_event("COMBAT_END", {"reason": reason})


def _end_combat_if_resolved(db: SessionDB) -> None:
    pcs = [a for a in db.list_actors("PC") if a.get("status") not in ("dead", "dying")]
    enemies = [a for a in db.list_actors("ENEMY") if a.get("status") not in ("dead", "dying")]
    if not pcs or not enemies:
        end_combat(db, "one_side_eliminated")


# ─────────────────────────────────────────────
# defender reaction / dice modifiers
# ─────────────────────────────────────────────

def choose_npc_reaction(db: SessionDB, attack: Dict[str, Any]) -> Dict[str, Any]:
    state = get_combat_state(db)
    target = _find_actor_by_name(db, attack.get("target_name", ""))
    if not target:
        return {"defender_option": "", "skill_name": "", "bonus_dice": 0, "penalty_dice": 0}

    attack_type = attack.get("action_type", "")
    target_idx = _participant_index(state, target["id"])
    participant = state["participants"][target_idx] if target_idx >= 0 else None

    if attack_type == "attack_firearm":
        option = "dive_for_cover" if _actor_skill(target, "Dodge") > 0 else ""
    elif attack_type in ("attack_melee", "maneuver"):
        dodge = _actor_skill(target, "Dodge")
        fight = _actor_skill(target, "Fighting (Brawl)")
        option = "fight_back" if fight >= dodge and fight > 0 else "dodge"
    else:
        option = ""

    skill_name = ""
    if option == "fight_back":
        skill_name = "Fighting (Brawl)"
    elif option in ("dodge", "dive_for_cover"):
        skill_name = "Dodge"

    bonus_dice = 0
    penalty_dice = 0

    # outnumbered: once a character has already responded this round,
    # subsequent melee attacks on them gain one bonus die.
    if participant and attack_type in ("attack_melee", "maneuver"):
        if int(participant.get("responses_this_round", 0)) >= 1:
            bonus_dice += 1

    return {
        "defender_option": option,
        "skill_name": skill_name,
        "bonus_dice": bonus_dice,
        "penalty_dice": penalty_dice,
    }


def _roll_defender_response(target: Dict[str, Any], defender_option: str, *, bonus_dice: int = 0, penalty_dice: int = 0) -> Dict[str, Any]:
    if not defender_option:
        return {
            "defender_roll": None,
            "defender_outcome": "",
            "defender_skill_name": "",
            "defender_skill_value": 0,
        }

    skill_name = "Fighting (Brawl)" if defender_option == "fight_back" else "Dodge"
    skill_value = _actor_skill(target, skill_name)
    if skill_value <= 0:
        return {
            "defender_roll": None,
            "defender_outcome": "failure",
            "defender_skill_name": skill_name,
            "defender_skill_value": 0,
        }

    roll = _roll_percentile(bonus_dice=bonus_dice, penalty_dice=penalty_dice)
    outcome = _percentile_success(roll, skill_value)
    return {
        "defender_roll": roll,
        "defender_outcome": outcome,
        "defender_skill_name": skill_name,
        "defender_skill_value": skill_value,
    }


# ─────────────────────────────────────────────
# damage and conditions
# ─────────────────────────────────────────────

def apply_major_wound_and_dying(db: SessionDB, actor_id: int, damage: int) -> Dict[str, Any]:
    actor = next((a for a in db.list_actors() if a["id"] == actor_id), None)
    if not actor:
        return {"major_wound": False, "unconscious": False, "dying": False, "dead": False, "status": "ok", "hp": None}

    cur_hp = int(actor.get("hp") or 0)
    max_hp = _calc_max_hp(actor)
    new_hp = max(0, cur_hp - max(0, int(damage)))

    major_wound = damage >= max(1, (max_hp + 1) // 2)
    unconscious = False
    dying = False
    dead = False
    status = actor.get("status", "ok")

    if damage >= max_hp:
        dead = True
        status = "dead"
    elif new_hp == 0:
        unconscious = True
        if major_wound:
            dying = True
            status = "dying"
        else:
            status = "unconscious"
    elif major_wound:
        con_value = _actor_stat(actor, "con", 50)
        con_roll = _roll_percentile()
        con_success = _percentile_success(con_roll, con_value)
        if _success_rank(con_success) <= 0:
            unconscious = True
            status = "unconscious"
        else:
            status = "injured"
    elif damage > 0 and status not in ("dead", "dying", "unconscious"):
        status = "injured"

    db.patch_actor(actor_id=actor_id, hp=new_hp, status=status)

    return {
        "major_wound": major_wound,
        "unconscious": unconscious,
        "dying": dying,
        "dead": dead,
        "status": status,
        "hp": new_hp,
    }


def apply_damage_and_conditions(db: SessionDB, actor_id: int, damage: int, *, source: str = "") -> Dict[str, Any]:
    res = apply_major_wound_and_dying(db, actor_id, damage)
    db.log_event("COMBAT_DAMAGE", {
        "actor_id": actor_id,
        "damage": damage,
        "source": source,
        "major_wound": res["major_wound"],
        "unconscious": res["unconscious"],
        "dying": res["dying"],
        "dead": res["dead"],
        "status": res["status"],
        "hp": res["hp"],
    })
    return res


def resolve_dying_checks_if_needed(db: SessionDB) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for actor in db.list_actors():
        if actor.get("status") != "dying":
            continue
        con_value = _actor_stat(actor, "con", 50)
        roll = _roll_percentile()
        outcome = _percentile_success(roll, con_value)
        if _success_rank(outcome) <= 0:
            db.patch_actor(actor_id=actor["id"], status="dead")
            results.append({
                "name": actor["name"],
                "roll": roll,
                "outcome": outcome,
                "status": "dead",
            })
        else:
            results.append({
                "name": actor["name"],
                "roll": roll,
                "outcome": outcome,
                "status": "dying",
            })
    return results


# ─────────────────────────────────────────────
# action submission and resolution
# ─────────────────────────────────────────────

def submit_combat_action(db: SessionDB, llm_result: Dict[str, Any]) -> Dict[str, Any]:
    state = get_combat_state(db)
    if not state.get("active"):
        return {"resolved": False, "reason": "combat_not_active"}

    action = llm_result.get("combat_action") or {}
    actor_name = (action.get("actor_name") or "").strip()
    actor = _find_actor_by_name(db, actor_name)
    if not actor:
        return {"resolved": False, "reason": f"actor_not_found:{actor_name}"}

    if actor["id"] != state.get("current_actor_id"):
        return {"resolved": False, "reason": "not_current_actor"}

    idx = _participant_index(state, actor["id"])
    if idx < 0:
        return {"resolved": False, "reason": "actor_not_in_combat"}

    participant = state["participants"][idx]

    if participant.get("forfeit_this_action"):
        participant["has_acted"] = True
        participant["forfeit_this_action"] = False
        state.setdefault("log", []).append({
            "type": "action_forfeited",
            "round": state["round"],
            "actor": actor["name"],
        })
        set_combat_state(db, state)
        advance_turn(db)
        return {
            "resolved": True,
            "combat_result": {
                "active": True,
                "round": state["round"],
                "actor_name": actor["name"],
                "action_type": "forfeit",
                "reason": "forfeit_previous_dive_for_cover",
            }
        }

    action_type = (action.get("action_type") or "").strip()

    if action_type in ("attack_melee", "maneuver"):
        res = resolve_melee_attack(db, action)
    elif action_type == "attack_firearm":
        res = resolve_firearm_attack(db, action)
    elif action_type == "dive_for_cover":
        participant["forfeit_next_action"] = True
        participant["has_acted"] = True
        state.setdefault("log", []).append({
            "type": "dive_for_cover_declared",
            "round": state["round"],
            "actor": actor["name"],
        })
        set_combat_state(db, state)
        advance_turn(db)
        res = {
            "resolved": True,
            "combat_result": {
                "active": True,
                "round": state["round"],
                "actor_name": actor["name"],
                "action_type": "dive_for_cover",
                "next_actor_name": (get_current_combat_actor(db) or {}).get("name", ""),
            }
        }
    elif action_type in ("move", "flee", "draw_weapon", "ready_firearm", "assist", "delay", "use_item", "cast", "other"):
        participant["has_acted"] = True
        if action_type == "ready_firearm":
            participant["readied_firearm"] = True
        state.setdefault("log", []).append({
            "type": "non_attack_action",
            "round": state["round"],
            "actor": actor["name"],
            "action_type": action_type,
            "target": action.get("target_name", ""),
        })
        set_combat_state(db, state)
        advance_turn(db)
        res = {
            "resolved": True,
            "combat_result": {
                "active": True,
                "round": state["round"],
                "actor_name": actor["name"],
                "action_type": action_type,
                "target_name": action.get("target_name", ""),
                "next_actor_name": (get_current_combat_actor(db) or {}).get("name", ""),
            }
        }
    else:
        res = {"resolved": False, "reason": f"unsupported_action:{action_type}"}

    _end_combat_if_resolved(db)
    return res


def resolve_melee_attack(db: SessionDB, action: Dict[str, Any]) -> Dict[str, Any]:
    state = get_combat_state(db)

    attacker = _find_actor_by_name(db, action.get("actor_name", ""))
    target = _find_actor_by_name(db, action.get("target_name", ""))

    if not attacker or not target:
        return {"resolved": False, "reason": "attacker_or_target_missing"}

    skill_name = action.get("skill_name") or "Fighting (Brawl)"
    skill_value = _actor_skill(attacker, skill_name)
    if skill_value <= 0:
        return {"resolved": False, "reason": f"missing_skill:{skill_name}"}

    defender_meta = choose_npc_reaction(db, {
        "target_name": target["name"],
        "action_type": action.get("action_type", "attack_melee"),
    })
    defender_option = defender_meta["defender_option"]

    attacker_roll = _roll_percentile()
    attacker_outcome = _percentile_success(attacker_roll, skill_value)

    defender = _roll_defender_response(
        target,
        defender_option,
        bonus_dice=defender_meta["bonus_dice"],
        penalty_dice=defender_meta["penalty_dice"],
    )

    a_rank = _success_rank(attacker_outcome)
    d_rank = _success_rank(defender["defender_outcome"])

    attacker_wins = False
    defender_hits_back = False

    if defender_option == "dodge":
        if a_rank > d_rank:
            attacker_wins = True
        elif a_rank == d_rank and a_rank > 0:
            attacker_wins = skill_value >= defender["defender_skill_value"]
    elif defender_option == "fight_back":
        if a_rank > d_rank:
            attacker_wins = True
        elif d_rank > a_rank:
            defender_hits_back = True
        elif a_rank > 0:
            attacker_wins = skill_value >= defender["defender_skill_value"]
            defender_hits_back = not attacker_wins
    else:
        attacker_wins = a_rank > 0

    weapon_name = action.get("weapon_name", "")
    weapon_damage = action.get("weapon_damage") or _default_weapon_damage(action.get("action_type", ""), weapon_name)
    damage_bonus = _calc_damage_bonus_expr(attacker)
    total_damage_expr = _combined_damage_expr(weapon_damage, damage_bonus)

    damage = 0
    target_update = None

    if attacker_wins and action.get("action_type") == "maneuver":
        pass
    elif attacker_wins:
        if attacker_outcome in ("extreme_success", "critical_success"):
            damage = _max_dice(total_damage_expr)
        else:
            damage = _roll_dice(total_damage_expr)
        target_update = apply_damage_and_conditions(
            db,
            target["id"],
            damage,
            source=f"{attacker['name']}:{skill_name}"
        )
    elif defender_hits_back:
        back_damage_expr = _combined_damage_expr("1D3", _calc_damage_bonus_expr(target))
        damage = _roll_dice(back_damage_expr)
        target_update = apply_damage_and_conditions(
            db,
            attacker["id"],
            damage,
            source=f"{target['name']}:fight_back"
        )

    atk_idx = _participant_index(state, attacker["id"])
    def_idx = _participant_index(state, target["id"])
    if atk_idx >= 0:
        state["participants"][atk_idx]["has_acted"] = True
    if def_idx >= 0:
        state["participants"][def_idx]["responses_this_round"] = int(state["participants"][def_idx].get("responses_this_round", 0)) + 1

    state.setdefault("log", []).append({
        "type": "melee_exchange",
        "round": state["round"],
        "attacker": attacker["name"],
        "target": target["name"],
        "attacker_roll": attacker_roll,
        "attacker_outcome": attacker_outcome,
        "defender_option": defender_option,
        "defender_roll": defender["defender_roll"],
        "defender_outcome": defender["defender_outcome"],
        "attacker_wins": attacker_wins,
        "defender_hits_back": defender_hits_back,
        "damage": damage,
    })
    set_combat_state(db, state)
    advance_turn(db)

    next_actor = get_current_combat_actor(db)
    return {
        "resolved": True,
        "combat_result": {
            "active": True,
            "round": state["round"],
            "actor_name": attacker["name"],
            "target_name": target["name"],
            "action_type": action.get("action_type", "attack_melee"),
            "skill_name": skill_name,
            "weapon_name": weapon_name,
            "attacker_roll": attacker_roll,
            "attacker_outcome": attacker_outcome,
            "defender_option": defender_option,
            "defender_roll": defender["defender_roll"],
            "defender_outcome": defender["defender_outcome"],
            "hit": attacker_wins and action.get("action_type") != "maneuver",
            "maneuver_success": attacker_wins and action.get("action_type") == "maneuver",
            "fight_back_hit": defender_hits_back,
            "damage": damage,
            "major_wound": (target_update or {}).get("major_wound", False),
            "unconscious": (target_update or {}).get("unconscious", False),
            "dying": (target_update or {}).get("dying", False),
            "dead": (target_update or {}).get("dead", False),
            "next_actor_name": next_actor["name"] if next_actor else "",
        }
    }


def resolve_firearm_attack(db: SessionDB, action: Dict[str, Any]) -> Dict[str, Any]:
    state = get_combat_state(db)

    attacker = _find_actor_by_name(db, action.get("actor_name", ""))
    target = _find_actor_by_name(db, action.get("target_name", ""))

    if not attacker or not target:
        return {"resolved": False, "reason": "attacker_or_target_missing"}

    skill_name = action.get("skill_name") or "Firearms (Handgun)"
    skill_value = _actor_skill(attacker, skill_name)
    if skill_value <= 0:
        return {"resolved": False, "reason": f"missing_skill:{skill_name}"}

    shots_fired = int(action.get("shots_fired", 0) or 1)
    range_band = (action.get("range_band") or "").strip().lower()

    penalty_dice = int(action.get("penalty_dice", 0) or 0)
    bonus_dice = int(action.get("bonus_dice", 0) or 0)

    if shots_fired in (2, 3) and "handgun" in skill_name.lower():
        penalty_dice += 1

    if range_band == "point_blank":
        bonus_dice += 1

    defender_option = "dive_for_cover" if _actor_skill(target, "Dodge") > 0 else ""
    defender_roll = None
    defender_outcome = ""

    if defender_option == "dive_for_cover":
        d_skill = _actor_skill(target, "Dodge")
        defender_roll = _roll_percentile()
        defender_outcome = _percentile_success(defender_roll, d_skill)
        if _success_rank(defender_outcome) > 0:
            penalty_dice += 1

    attacker_roll = _roll_percentile(bonus_dice=bonus_dice, penalty_dice=penalty_dice)
    attacker_outcome = _percentile_success(attacker_roll, skill_value)

    hit = False
    a_rank = _success_rank(attacker_outcome)
    d_rank = _success_rank(defender_outcome)

    if defender_option == "dive_for_cover":
        if a_rank > d_rank:
            hit = True
        elif a_rank == d_rank and a_rank > 0:
            hit = skill_value >= _actor_skill(target, "Dodge")
    else:
        hit = a_rank > 0

    weapon_name = action.get("weapon_name", "")
    weapon_damage = action.get("weapon_damage") or _default_weapon_damage("attack_firearm", weapon_name)

    damage = 0
    target_update = None
    if hit:
        if attacker_outcome in ("extreme_success", "critical_success"):
            damage = _max_dice(weapon_damage)
        else:
            damage = _roll_dice(weapon_damage)
        target_update = apply_damage_and_conditions(
            db,
            target["id"],
            damage,
            source=f"{attacker['name']}:{skill_name}"
        )

    atk_idx = _participant_index(state, attacker["id"])
    if atk_idx >= 0:
        state["participants"][atk_idx]["has_acted"] = True

    def_idx = _participant_index(state, target["id"])
    if def_idx >= 0:
        if defender_option == "dive_for_cover":
            state["participants"][def_idx]["responses_this_round"] = int(state["participants"][def_idx].get("responses_this_round", 0)) + 1
            state["participants"][def_idx]["forfeit_next_action"] = True

    state.setdefault("log", []).append({
        "type": "firearm_attack",
        "round": state["round"],
        "attacker": attacker["name"],
        "target": target["name"],
        "attacker_roll": attacker_roll,
        "attacker_outcome": attacker_outcome,
        "defender_option": defender_option,
        "defender_roll": defender_roll,
        "defender_outcome": defender_outcome,
        "damage": damage,
        "hit": hit,
    })
    set_combat_state(db, state)
    advance_turn(db)

    next_actor = get_current_combat_actor(db)
    return {
        "resolved": True,
        "combat_result": {
            "active": True,
            "round": state["round"],
            "actor_name": attacker["name"],
            "target_name": target["name"],
            "action_type": "attack_firearm",
            "skill_name": skill_name,
            "weapon_name": weapon_name,
            "shots_fired": shots_fired,
            "range_band": range_band,
            "attacker_roll": attacker_roll,
            "attacker_outcome": attacker_outcome,
            "defender_option": defender_option,
            "defender_roll": defender_roll,
            "defender_outcome": defender_outcome,
            "hit": hit,
            "damage": damage,
            "major_wound": (target_update or {}).get("major_wound", False),
            "unconscious": (target_update or {}).get("unconscious", False),
            "dying": (target_update or {}).get("dying", False),
            "dead": (target_update or {}).get("dead", False),
            "next_actor_name": next_actor["name"] if next_actor else "",
        }
    }


def resolve_maneuver(db: SessionDB, action: Dict[str, Any]) -> Dict[str, Any]:
    action = dict(action)
    action["action_type"] = "maneuver"
    return resolve_melee_attack(db, action)


# ─────────────────────────────────────────────
# top-level orchestration
# ─────────────────────────────────────────────

def resolve_combat_turn(db: SessionDB, llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for engine.py.

    Expected llm_result keys:
      - combat_action (new contract)
    Mutates/returns llm_result with:
      - combat_result
      - updated_actor (when damage/status changed)
    """
    maybe_start_combat(db, llm_result)

    state = get_combat_state(db)
    if not state.get("active"):
        return llm_result

    combat_action = llm_result.get("combat_action") or {}
    if not combat_action:
        return llm_result

    res = submit_combat_action(db, llm_result)
    if res.get("resolved"):
        llm_result["combat_result"] = res.get("combat_result", {})
        combat_result = llm_result["combat_result"]

        target_name = combat_result.get("target_name", "")
        if target_name and combat_result.get("damage", 0) > 0:
            target = _find_actor_by_name(db, target_name)
            if target:
                llm_result["updated_actor"] = {
                    "name": target["name"],
                    "hp": target.get("hp"),
                    "status": target.get("status", "ok"),
                }

    dying_checks = resolve_dying_checks_if_needed(db)
    if dying_checks:
        llm_result.setdefault("combat_result", {})
        llm_result["combat_result"]["dying_checks"] = dying_checks

    _end_combat_if_resolved(db)
    llm_result["combat_state"] = get_combat_state(db)
    return llm_result