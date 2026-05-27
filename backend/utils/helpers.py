import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# pylint: disable=import-error
from utils.db_session import SessionDB
from utils.schemas import validate_chat_response_payload

CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_CANDIDATE_DIRS = [
    CURRENT_DIR / "prompts",
    CURRENT_DIR.parent / "prompts",
    Path("/mnt/data/prompts"),
    CURRENT_DIR,
    Path("/mnt/data"),
]

logger = logging.getLogger("keeper_ai.helpers")

COC_SUCCESS_RANK = {
    "fumble": -1,
    "failure": 0,
    "regular_success": 1,
    "hard_success": 2,
    "extreme_success": 3,
    "critical_success": 4,
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


def coc_percentile_success(roll: int, value: int) -> str:
    value = max(0, int(value or 0))
    roll = int(roll or 0)

    if roll <= 0:
        return "failure"
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


def coc_success_rank(outcome: str) -> int:
    return COC_SUCCESS_RANK.get((outcome or "").strip().lower(), 0)


def parse_roll_resolution_from_message(message: str) -> dict | None:
    text = str(message or "").strip()
    if not text:
        return None

    used_luck = False
    luck_spent = 0
    m_luck = re.search(
        r"spent\s+(\d+)\s+Luck|витратив(?:ла)?\s+(\d+)\s+Luck|потратил(?:а)?\s+(\d+)\s+Luck",
        text,
        flags=re.IGNORECASE,
    )
    if m_luck:
        used_luck = True
        for grp in m_luck.groups():
            if grp:
                luck_spent = int(grp)
                break

    m = re.search(r"🎲\s*(.*?)\s*\((.*?)\)\s*:\s*(\d{1,3})\s*/\s*(\d{1,3})", text, flags=re.DOTALL)
    if not m:
        m = re.search(
            r"(?:investigator|actor|character)\s*[:=]\s*(.*?)\s*(?:\n|,|;).*?"
            r"(?:skill)\s*[:=]\s*(.*?)\s*(?:\n|,|;).*?"
            r"(?:roll)\s*[:=]\s*(\d{1,3}).*?"
            r"(?:target|value)\s*[:=]\s*(\d{1,3})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if not m:
        return None

    investigator = str(m.group(1) or "").strip()
    skill = str(m.group(2) or "").strip()
    roll = int(m.group(3))
    target = int(m.group(4))

    return {
        "investigator": investigator,
        "skill": skill,
        "target": target,
        "roll": roll,
        "outcome": coc_percentile_success(roll, target),
        "raw_verdict": text[:1000],
        "roll_type": "skill_check",
        "used_luck": used_luck,
        "luck_spent": luck_spent,
    }


def _blank_roll_request() -> dict:
    return {"required": False, "skill_name": "", "action_text": "", "reason": ""}


def _detect_roll_request_from_suggested_actions(suggested_actions: list[str] | None) -> dict | None:
    for action in suggested_actions or []:
        m = re.match(r"^(.*?)\s*→\s*Roll\s+(.+?)\s*$", (action or "").strip(), re.IGNORECASE)
        if not m:
            continue
        return {
            "required": True,
            "action_text": m.group(1).strip(),
            "skill_name": m.group(2).strip(),
            "reason": "This action was already established as requiring a roll.",
        }
    return None


def _normalize_repaired_blueprint(bp: dict, act_count: int, scene_counts: list[int], era_context: str, seed: str, plan: dict) -> dict:
    bp = dict(bp or {})
    bp.setdefault("title", "Untitled Scenario")
    bp.setdefault("era_and_setting", era_context)
    bp.setdefault("atmosphere_notes", "Tense and uncanny.")
    bp.setdefault("inciting_hook", (seed or "")[:220])
    bp.setdefault("core_mystery", "Something hidden is distorting the visible pattern.")
    bp.setdefault("hidden_threat", "A hidden force is exploiting the situation.")
    bp.setdefault("truth_the_players_never_suspect", "The anomaly is active, strategic, and not merely environmental.")
    bp.setdefault("scenario_engine", {
        "surface_explanation": "A mundane explanation seems plausible at first.",
        "actual_explanation": "The visible pattern conceals an intentional mechanism.",
        "false_leads": (plan or {}).get("false_leads", ["", ""])[:2],
        "contradictions": (plan or {}).get("contradictions", ["", ""])[:2],
        "reversals": (plan or {}).get("reversals", ["", ""])[:2],
        "dynamic_pressures": (plan or {}).get("dynamic_pressures", ["", ""])[:2],
        "climax_choices": [
            {"option": "Contain the threat", "cost": "Immediate personal loss", "consequence": "The danger is reduced, but the cost remains."},
            {"option": "Expose the truth", "cost": "Public fallout", "consequence": "The pattern becomes visible, but retaliation follows."},
        ],
    })

    acts = bp.get("acts") if isinstance(bp.get("acts"), list) else []
    while len(acts) < act_count:
        acts.append({
            "act": len(acts) + 1,
            "title": f"Act {len(acts) + 1}",
            "summary": "Investigation deepens.",
            "purpose": "Advance the scenario.",
            "belief_shift": "The obvious explanation weakens.",
            "required_payoffs": ["evidence", "pressure"],
            "scenes": [],
        })
    acts = acts[:act_count]

    for i, expected in enumerate(scene_counts):
        act = acts[i]
        act.setdefault("act", i + 1)
        act.setdefault("title", f"Act {i + 1}")
        act.setdefault("summary", "Investigation deepens.")
        act.setdefault("purpose", "Advance the scenario.")
        act.setdefault("belief_shift", "The old theory no longer holds.")
        act.setdefault("required_payoffs", ["evidence", "pressure"])
        scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
        while len(scenes) < expected:
            scenes.append({
                "scene": f"Scene {len(scenes) + 1}",
                "location": "Unfixed location",
                "scene_function": "investigation",
                "dramatic_question": "What does this scene reveal?",
                "entry_condition": "Investigators follow the current lead.",
                "exit_condition": "They leave with a clearer direction.",
                "trigger": "A clue or pressure forces movement.",
                "description": "The investigators confront a new layer of the problem.",
                "what_happens": "Evidence and pressure push the situation forward.",
                "pressure_if_delayed": "The situation worsens.",
                "reveals": [],
                "conceals": [],
                "clues_available": [],
                "npc_present": [],
                "threat_level": "tension",
                "keeper_notes": "Keep the scene concise and actionable.",
            })
        act["scenes"] = scenes[:expected]
        acts[i] = act

    bp["acts"] = acts
    bp.setdefault("locations", [])
    bp.setdefault("npcs", [])
    bp.setdefault("clues", [])
    bp.setdefault("plot_threads", [])
    return bp


def read_prompt(filename: str, *, prompt_dir: str | None = None) -> str:
    rel = Path(filename)
    if prompt_dir:
        candidate = Path(prompt_dir) / rel
        if not candidate.exists():
            raise FileNotFoundError(f"Prompt missing: {candidate}")
        return candidate.read_text(encoding="utf-8")

    for root in PROMPT_CANDIDATE_DIRS:
        candidate = root / rel
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: {filename}")


def detect_degenerate_output(text: str, threshold: int = 30) -> bool:
    if not text or len(text) < 200:
        return False
    tokens = text.split()
    if len(tokens) < threshold:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i].strip(".,;:") == tokens[i - 1].strip(".,;:"):
            run += 1
            if run >= threshold:
                return True
        else:
            run = 1
    return False


def clean_degenerate_value(value: str, max_len: int = 1200) -> str:
    if not value:
        return ""
    cleaned = []
    prev = None
    repeats = 0
    for token in value.split():
        norm = token.strip(".,;:")
        if norm == prev:
            repeats += 1
            if repeats >= 3:
                continue
        else:
            repeats = 0
        prev = norm
        cleaned.append(token)
    return " ".join(cleaned).strip()[:max_len]


# Compatibility shims: runtime localization now lives in prompt_translate.py only.
def localize_blueprint_fields(blueprint: dict, language: str) -> dict:
    return blueprint if isinstance(blueprint, dict) else {}


def localize_opening_result_fields(result: dict, language: str) -> dict:
    return dict(result or {})


def _doc_setting_signature(doc) -> tuple[str, str]:
    meta = getattr(doc, "metadata", {}) or {}
    blob = " ".join(
        str(meta.get(k, "") or "").strip().lower()
        for k in ("archetype", "role", "type", "title_en", "abstraction")
        if str(meta.get(k, "") or "").strip()
    )

    setting = "generic"
    if any(k in blob for k in ("cyber", "server", "neural", "megacorp", "ar", "drone", "lab")):
        setting = "cyberpunk"
    elif any(k in blob for k in ("expedition", "ruin", "archae", "desert", "mountain", "monastery", "jungle")):
        setting = "expedition"
    elif any(k in blob for k in ("ship", "port", "ocean", "lighthouse", "trench", "nautical")):
        setting = "nautical"
    elif any(k in blob for k in ("space", "station", "colony", "void", "relay")):
        setting = "space"
    return setting, blob


def _sanitize_blueprint_candidate(candidate: str) -> str:
    candidate = (candidate or "").strip().replace("<%", "").replace("%>", "")
    candidate = re.sub(r'"dynamic_press[^\"]*pressures"\s*:', '"dynamic_pressures":', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'"exit_[a-zA-Z]+_condition"\s*:', '"exit_condition":', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r',\s*/\s*(?=\")', ', ', candidate)
    candidate = re.sub(r'(?m)^\s*/\s*$', '', candidate)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", candidate)
    candidate = re.sub(r"(?m)^\s*//.*$", "", candidate)
    candidate = re.sub(r"(?m)^\s*#.*$", "", candidate)
    candidate = re.sub(r"(?m),\s*(//.*)?$", ",", candidate)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = re.sub(r'(?m)^(\s*)(?:та|і|й|и|and)\s+(?=")', r"\1", candidate)
    return candidate


def extract_blueprint_json(text: str) -> dict:
    raw_text = (text or "").strip()
    logger.info("extract_blueprint_json(): raw preview: %r", raw_text[:4000])
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", raw_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No complete JSON object found in blueprint response")

    candidate = _sanitize_blueprint_candidate(cleaned[start:end + 1])
    logger.info("extract_blueprint_json(): candidate preview: %r", candidate[:4000])
    if detect_degenerate_output(raw_text):
        raise ValueError("Degenerate repetitive output detected — model likely hit token limit")

    parsed = json.loads(candidate, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("Blueprint response is not a JSON object")

    required = ["title", "era_and_setting", "inciting_hook", "acts"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"Blueprint missing required keys: {missing}")
    if not isinstance(parsed.get("acts"), list) or not parsed["acts"]:
        raise ValueError("Blueprint acts must be a non-empty list")
    if not str(parsed.get("title", "") or "").strip():
        raise ValueError("Blueprint title is empty")
    if not str(parsed.get("inciting_hook", "") or "").strip():
        raise ValueError("Blueprint inciting_hook is empty")
    return parsed


def _normalize_legacy_contract_fields(parsed: dict) -> dict:
    safe = dict(parsed or {})
    if "combat_action" not in safe and isinstance(safe.get("combat_update"), dict):
        cu = dict(safe.get("combat_update") or {})
        safe["combat_action"] = {
            "start_combat": bool(cu.get("combat_started", False)),
            "end_combat": bool(cu.get("combat_ended", False)),
            "actor_name": str(cu.get("attacker_name", "") or ""),
            "target_name": str(cu.get("target_name", "") or ""),
            "action_type": str(cu.get("attack_mode", "") or ""),
            "skill_name": str(cu.get("skill_name", "") or ""),
            "weapon_name": str(cu.get("weapon_name", "") or ""),
            "weapon_damage": str(cu.get("weapon_damage", "") or ""),
            "defender_option": str(cu.get("defender_option", "") or ""),
            "shots_fired": int(cu.get("shots_fired", 0) or 0),
            "bonus_dice": int(cu.get("bonus_dice", 0) or 0),
            "penalty_dice": int(cu.get("penalty_dice", 0) or 0),
        }
    safe.pop("combat_update", None)
    return safe


def _extract_json_candidate(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    tagged = re.search(r"<SYSTEM_RESPONSE_JSON>\s*(\{.*?\})\s*</SYSTEM_RESPONSE_JSON>", text, re.DOTALL)
    if tagged:
        logger.info("extract_json(): found SYSTEM_RESPONSE_JSON envelope")
        return tagged.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _fallback_chat_payload(text: str) -> dict:
    narrative_match = re.search(r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL | re.IGNORECASE)
    suggested_match = re.search(r'"suggested_actions"\s*:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)

    narrative = ""
    if narrative_match:
        narrative = narrative_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

    suggested_actions: list[str] = []
    if suggested_match:
        raw_items = re.findall(r'"((?:[^"\\]|\\.)*)"', suggested_match.group(1), re.DOTALL)
        suggested_actions = [x.replace("\\n", " ").replace('\\"', '"').strip() for x in raw_items if x.strip()][:3]

    action_block_match = re.search(r"^(.*?)(?:Suggested actions?)\s*:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if not narrative and action_block_match:
        narrative = action_block_match.group(1).strip()
        for line in action_block_match.group(2).splitlines():
            clean = re.sub(r"^[\-\*\•\d\.\)\s]+", "", line.strip())
            if clean:
                suggested_actions.append(clean)
        suggested_actions = suggested_actions[:3]

    if not narrative:
        narrative = text.strip()

    return {
        "narrative": narrative,
        "suggested_actions": suggested_actions,
        "state_updates": dict(_EMPTY_STATE_UPDATES),
        "roll_resolution": None,
        "roll_request": _detect_roll_request_from_suggested_actions(suggested_actions) or _blank_roll_request(),
        "combat_action": dict(_EMPTY_COMBAT_ACTION),
        "scene_entities": {"present_named_entities": []},
        "image_url": None,
        "generation_id": None,
    }


def extract_json(text: str) -> dict:
    raw_text = text or ""
    logger.info("extract_json(): raw response preview: %r", raw_text[:2000])
    candidate = _extract_json_candidate(raw_text)
    logger.info("extract_json(): normalized candidate preview: %r", candidate[:2000])
    candidate = re.sub(r":\s*([-+]?\d+)[dD]\d+", r": \1", candidate)

    try:
        parsed = json.loads(candidate, strict=False)
        if isinstance(parsed, dict):
            parsed = _normalize_legacy_contract_fields(parsed)
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
            return validate_chat_response_payload(parsed)
    except json.JSONDecodeError as exc:
        logger.warning("extract_json(): JSON parse error: %s", exc)
        logger.warning("extract_json(): failed candidate body: %r", candidate[:4000])

    recovered = _fallback_chat_payload(candidate)
    logger.warning("extract_json(): recovered fallback payload: %r", recovered)
    return validate_chat_response_payload(recovered)


def get_chat_history(db: SessionDB, limit: int = 10) -> str:
    events = db.list_events(limit=max(limit * 3, 24))
    pairs: list[tuple[str, str]] = []
    for event in events:
        if event.get("event_type") != "CHAT":
            continue
        payload = event.get("payload", {}) or {}
        role = str(payload.get("role", "Unknown") or "Unknown")
        content = str(payload.get("content", "") or "").strip()
        if content:
            pairs.append((role, content))

    recent_pairs = pairs[-max(6, min(limit, 12)):]
    lines = []
    for role, content in recent_pairs:
        if role.strip().lower() == "keeper":
            if len(content) > 450:
                content = f"{content[:180].strip()} … {content[-180:].strip()}"
            lines.append(f"KEEPER: {content}")
        else:
            if len(content) > 280:
                content = content[:280].strip()
            lines.append(f"USER: {content}")
    return "\n\n".join(lines)


def kv_get(cur, key: str, default: str = "") -> str:
    cur.execute("SELECT value FROM kv_store WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


def kv_set(db: SessionDB, key: str, value: str) -> None:
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", (key, value))
    db.conn.commit()


def is_roll_verdict_message(message: str) -> bool:
    text = message or ""
    return "VERDICT" in text and ("[SYSTEM MESSAGE" in text or "ROLL_VERDICT:" in text)


def has_roll_verdict(message: str) -> bool:
    return is_roll_verdict_message(message)


def build_verdict_guard(message: str) -> str:
    if not is_roll_verdict_message(message):
        return ""
    return (
        "\n\n⚠ VERDICT RECEIVED. STRICT RULES FOR THIS RESPONSE:\n"
        "- Write 1–3 sentences MAXIMUM.\n"
        "- Describe ONLY what changed due to this roll result.\n"
        "- DO NOT reproduce or paraphrase any prior narrative.\n"
        "- DO NOT re-describe the location, NPCs, or atmosphere.\n"
        "- Start mid-action, from the moment the roll resolves.\n"
    )


def extract_last_turn_ban(db: SessionDB) -> str:
    last_keeper_events = [
        e for e in db.list_events(limit=6)
        if e.get("event_type") == "CHAT" and e.get("payload", {}).get("role") == "Keeper"
    ]
    if not last_keeper_events:
        return ""
    last_narrative = str(last_keeper_events[0].get("payload", {}).get("content", "") or "")
    sentences = [s.strip() for s in re.split(r"[.!?。]", last_narrative) if len(s.strip()) > 20]
    if not sentences:
        return ""
    return "PHRASES FROM YOUR PREVIOUS RESPONSE (DO NOT REUSE OR PARAPHRASE THESE):\n" + "\n".join(
        f'- "{s[:80]}"' for s in sentences[:5]
    )


def _safe_json_loads(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _ollama_task_defaults(task: str | None) -> dict:
    task = (task or "default").strip().lower()
    profiles = {
        "scenario_json": ("OLLAMA_MODEL_SCENARIO_JSON", "gemma3:27b", True, 16384, 2200, 1.10),
        "character_json": ("OLLAMA_MODEL_CHARACTER_JSON", "gemma3:27b", True, 8192, 1800, 1.10),
        "opening_json": ("OLLAMA_MODEL_OPENING_JSON", "gemma3:27b", True, 8192, 1600, 1.10),
        "scenario_tagged": ("OLLAMA_MODEL_SCENARIO_TAGGED", "gemma3:27b", False, 12000, 1800, 1.05),
        "chat_text": ("OLLAMA_MODEL_CHAT_TEXT", "gemma3:27b", False, 16384, 1800, 1.05),
        "translation_text": ("OLLAMA_MODEL_TRANSLATION", "gemma3:27b", False, 8192, 1800, 1.00),
        "default": ("OLLAMA_MODEL", "gemma3:27b", False, 8192, 1600, 1.10),
    }
    env_model, default_model, default_json, default_ctx, default_predict, default_repeat = profiles.get(task, profiles["default"])
    env_prefix = task.upper()
    return {
        "model": os.getenv(env_model, default_model),
        "json_mode": default_json,
        "num_ctx": _int_env(f"{env_prefix}_NUM_CTX", default_ctx),
        "num_predict": _int_env(f"{env_prefix}_NUM_PREDICT", default_predict),
        "repeat_penalty": _float_env(f"{env_prefix}_REPEAT_PENALTY", default_repeat),
    }


def _openai_task_defaults(task: str | None) -> dict:
    task = (task or "default").strip().lower()
    env_key = f"OPENAI_MODEL_{task.upper()}"
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Allow cheap model routing without touching code.
    # Example:
    # OPENAI_MODEL_TRANSLATION_TEXT=gpt-4o-mini
    # OPENAI_MODEL_CHAT_TEXT=gpt-4o
    model = os.getenv(env_key, default_model)

    return {
        "model": model,
        "max_tokens": _int_env(f"OPENAI_{task.upper()}_MAX_TOKENS", 0),
    }


def get_llm(
    temperature: float = 0.7,
    *,
    task: str | None = None,
    streaming: bool = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    json_mode: bool | None = None,
    json_schema: dict | None = None,
    repeat_penalty: float | None = None,
    model: str | None = None,
):
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    task_name = (task or "default").strip().lower()

    if provider == "openai":
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        defaults = _openai_task_defaults(task_name)
        openai_model = model or defaults["model"]
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        kwargs = {
            "model": openai_model,
            "temperature": temperature,
            "api_key": api_key,
            "streaming": streaming,
        }
        if defaults.get("max_tokens", 0) > 0:
            kwargs["max_tokens"] = defaults["max_tokens"]

        logger.info("LLM_FACTORY provider=openai task=%s model=%s streaming=%s", task_name, openai_model, streaming)
        return ChatOpenAI(**kwargs) | StrOutputParser()

    from langchain_ollama import OllamaLLM

    profile = _ollama_task_defaults(task_name)
    resolved_model = model or profile["model"]
    resolved_num_ctx = int(num_ctx) if num_ctx is not None else profile["num_ctx"]
    resolved_num_predict = int(num_predict) if num_predict is not None else profile["num_predict"]
    resolved_json_mode = profile["json_mode"] if json_mode is None else bool(json_mode)
    resolved_repeat_penalty = float(repeat_penalty) if repeat_penalty is not None else profile["repeat_penalty"]

    kwargs = {
        "model": resolved_model,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": temperature,
        "num_ctx": resolved_num_ctx,
        "num_predict": resolved_num_predict,
        "repeat_penalty": resolved_repeat_penalty,
    }
    if json_schema is not None:
        kwargs["format"] = json_schema
    elif resolved_json_mode:
        kwargs["format"] = "json"

    logger.info("LLM_FACTORY provider=ollama task=%s model=%s json=%s", task_name, resolved_model, bool(kwargs.get("format")))
    return OllamaLLM(**kwargs)
