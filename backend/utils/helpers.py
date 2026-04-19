import json
import logging
import os
import re
from pathlib import Path
#pylint: disable=import-error
from utils.db_session import SessionDB

CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_CANDIDATE_DIRS = [
    CURRENT_DIR / "prompts",
    CURRENT_DIR.parent / "prompts",
    Path("/mnt/data/prompts"),
    CURRENT_DIR,
    Path("/mnt/data"),
]

logger = logging.getLogger("keeper_ai.helpers")


# LANGUAGE_NAMES = {
#     "ua": "Ukrainian",
#     "en": "English",
#     "hr": "Croatian",
#     "ru": "Russian",
#     "pl": "Polish",
#     "de": "German",
#     "fr": "French",
#     "es": "Spanish",
#     "it": "Italian",
#     "zh": "Chinese",
#     "ja": "Japanese",
# }


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
    bp.setdefault("inciting_hook", seed[:220])
    bp.setdefault("core_mystery", "Something hidden is draining people and distorting the visible pattern.")
    bp.setdefault("hidden_threat", "A hidden force is exploiting dream logic to extract identity and agency.")
    bp.setdefault("truth_the_players_never_suspect", "The theft changes both victims and the thief.")

    bp.setdefault("scenario_engine", {
        "surface_explanation": "A mundane explanation seems plausible at first.",
        "actual_explanation": "The visible pattern conceals an intentional predatory mechanism.",
        "false_leads": plan.get("false_leads", ["", ""])[:2],
        "contradictions": plan.get("contradictions", ["", ""])[:2],
        "reversals": plan.get("reversals", ["", ""])[:2],
        "dynamic_pressures": plan.get("dynamic_pressures", ["", ""])[:2],
        "climax_choices": [
            {"option": "Contain the threat", "cost": "Immediate personal loss", "consequence": "The damage stops but the cost remains."},
            {"option": "Expose the truth", "cost": "Public panic and retaliation", "consequence": "The pattern becomes visible to everyone."},
        ],
    })

    acts = bp.get("acts")
    if not isinstance(acts, list):
        acts = []

    while len(acts) < act_count:
        acts.append({
            "act": len(acts) + 1,
            "title": f"Act {len(acts) + 1}",
            "summary": "Investigation deepens.",
            "purpose": "Move toward understanding.",
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

        scenes = act.get("scenes")
        if not isinstance(scenes, list):
            scenes = []

        while len(scenes) < expected:
            scenes.append({
                "scene": f"Scene {len(scenes) + 1}",
                "location": "Unfixed location",
                "scene_function": "investigation",
                "dramatic_question": "What does this scene reveal?",
                "entry_condition": "Investigators follow the current lead.",
                "exit_condition": "They gain a clearer direction.",
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
            raise FileNotFoundError(f"Translated prompt missing: {candidate}")
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
        if tokens[i].strip('.,;:') == tokens[i-1].strip('.,;:'):
            run += 1
            if run >= threshold:
                return True
        else:
            run = 1
    return False


def clean_degenerate_value(value: str, max_len: int = 1200) -> str:
    if not value:
        return ""
    tokens = value.split()
    cleaned = []
    prev = None
    repeats = 0
    for t in tokens:
        norm = t.strip('.,;:')
        if norm == prev:
            repeats += 1
            if repeats >= 3:
                continue
        else:
            repeats = 0
        prev = norm
        cleaned.append(t)
    result = " ".join(cleaned).strip()
    return result[:max_len]

# Intentionally no LLM-based localization in start-session/opening path.
# Keep these as compatibility shims so existing imports do not break.
def localize_blueprint_fields(blueprint: dict, language: str) -> dict:
    return blueprint if isinstance(blueprint, dict) else {}


def localize_opening_result_fields(result: dict, language: str) -> dict:
    return dict(result or {})


def _doc_setting_signature(doc) -> tuple[str, str]:
    meta = getattr(doc, "metadata", {}) or {}

    archetype = str(meta.get("archetype", "") or "").strip().lower()
    role = str(meta.get("role", "") or "").strip().lower()
    atom_type = str(meta.get("type", "") or "").strip().lower()
    title = str(meta.get("title_en", "") or "").strip().lower()
    abstraction = str(meta.get("abstraction", "") or "").strip().lower()

    blob = " ".join(x for x in [archetype, role, atom_type, title, abstraction] if x)

    setting = "generic"
    if any(k in blob for k in ["cyber", "server", "neural", "megacorp", "ar", "drone", "lab"]):
        setting = "cyberpunk"
    elif any(k in blob for k in ["expedition", "ruin", "archae", "desert", "mountain", "monastery", "jungle"]):
        setting = "expedition"
    elif any(k in blob for k in ["ship", "port", "ocean", "lighthouse", "trench", "nautical"]):
        setting = "nautical"
    elif any(k in blob for k in ["space", "station", "colony", "void", "relay"]):
        setting = "space"

    return setting, blob


def _sanitize_blueprint_candidate(candidate: str) -> str:
    candidate = (candidate or "").strip()
    candidate = candidate.replace("<%", "").replace("%>", "")

    candidate = re.sub(r'"dynamic_press[^\"]*pressures"\s*:', '"dynamic_pressures":', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'"exit_[a-zA-Z]+_condition"\s*:', '"exit_condition":', candidate, flags=re.IGNORECASE)

    candidate = re.sub(r',\s*/\s*(?=\")', ', ', candidate)
    candidate = re.sub(r'(?m)^\s*/\s*$', '', candidate)
    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
    candidate = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", candidate)
    candidate = re.sub(r'(?m)^\s*//.*$', '', candidate)
    candidate = re.sub(r'(?m)^\s*#.*$', '', candidate)
    candidate = re.sub(r'(?m),\s*(//.*)?$', ',', candidate)
    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
    candidate = re.sub(r'(?m)^(\s*)(?:та|і|й|и|and)\s+(?=")', r'\1', candidate)

    candidate = re.sub(
        r'(?m)^\s*(?![")\]\},\{\[])(?!true\b)(?!false\b)(?!null\b)[A-Za-zА-Яа-яІіЇїЄєҐґ].*$',
        '',
        candidate,
    )
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
    
    try:
        parsed = json.loads(candidate, strict=False)
    except json.JSONDecodeError as e:
        logger.warning("extract_blueprint_json(): JSON parse error: %s", e)
        logger.warning("extract_blueprint_json(): failed candidate body: %r", candidate[:4000])
        raise

    if not isinstance(parsed, dict):
        raise ValueError("Blueprint response is not a JSON object")

    required = ["title", "era_and_setting", "inciting_hook", "acts"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"Blueprint missing required keys: {missing}")

    acts = parsed.get("acts")
    if not isinstance(acts, list) or not acts:
        raise ValueError("Blueprint acts must be a non-empty list")

    if not str(parsed.get("title", "") or "").strip():
        raise ValueError("Blueprint title is empty")

    if not str(parsed.get("inciting_hook", "") or "").strip():
        raise ValueError("Blueprint inciting_hook is empty")

    return parsed


def extract_json(text: str) -> dict:
    raw_text = text or ""
    logger.info("extract_json(): raw response preview: %r", raw_text[:2000])

    text = raw_text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)

    tagged = re.search(r"<SYSTEM_RESPONSE_JSON>\s*(\{.*?\})\s*</SYSTEM_RESPONSE_JSON>", text, re.DOTALL)
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

    narrative_match = re.search(r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL | re.IGNORECASE)
    suggested_match = re.search(r'"suggested_actions"\s*:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)

    narrative = ""
    if narrative_match:
        narrative = narrative_match.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

    suggested_actions: list[str] = []
    if suggested_match:
        raw_items = re.findall(r'"((?:[^"\\]|\\.)*)"', suggested_match.group(1), re.DOTALL)
        suggested_actions = [
            item.replace("\\n", " ").replace('\\"', '"').strip()
            for item in raw_items
            if item.strip()
        ][:3]

    action_block_match = re.search(r"^(.*?)(?:Suggested actions?)\s*:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
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


def get_chat_history(db: SessionDB, limit: int = 10) -> str:
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


def kv_get(cur, key: str, default: str = "") -> str:
    cur.execute("SELECT value FROM kv_store WHERE key=?", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


def kv_set(db: SessionDB, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
        (key, value),
    )
    db.conn.commit()


def is_roll_verdict_message(message: str) -> bool:
    text = message or ""
    return "VERDICT" in text and ("[SYSTEM MESSAGE" in text or "ROLL_VERDICT:" in text)


def has_roll_verdict(message: str) -> bool:
    return is_roll_verdict_message(message)


def build_verdict_guard(message: str) -> str:
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


def extract_last_turn_ban(db: SessionDB) -> str:
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


def _safe_json_loads(value: str | None):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())




def _ollama_task_defaults(task: str | None) -> dict:
    task = (task or "default").strip().lower()

    profiles = {
        # strict structured generation
        "scenario_json": {
            "model": os.getenv("OLLAMA_MODEL_SCENARIO_JSON", "gemma3:27b"),
            "json_mode": True,
            "num_ctx": int(os.getenv("SCENARIO_JSON_NUM_CTX", "16384")),
            "num_predict": int(os.getenv("SCENARIO_JSON_NUM_PREDICT", "2200")),
            "repeat_penalty": float(os.getenv("SCENARIO_JSON_REPEAT_PENALTY", "1.10")),
        },
        "character_json": {
            "model": os.getenv("OLLAMA_MODEL_CHARACTER_JSON", "gemma3:27b"),
            "json_mode": True,
            "num_ctx": int(os.getenv("CHARACTER_JSON_NUM_CTX", "8192")),
            "num_predict": int(os.getenv("CHARACTER_JSON_NUM_PREDICT", "1800")),
            "repeat_penalty": float(os.getenv("CHARACTER_JSON_REPEAT_PENALTY", "1.10")),
        },
        "opening_json": {
            "model": os.getenv("OLLAMA_MODEL_OPENING_JSON", "gemma3:27b"),
            "json_mode": True,
            "num_ctx": int(os.getenv("OPENING_JSON_NUM_CTX", "8192")),
            "num_predict": int(os.getenv("OPENING_JSON_NUM_PREDICT", "1600")),
            "repeat_penalty": float(os.getenv("OPENING_JSON_REPEAT_PENALTY", "1.10")),
        },

        # tagged/plain-text generation
        "scenario_tagged": {
            "model": os.getenv("OLLAMA_MODEL_SCENARIO_TAGGED", "gemma3:27b"),
            "json_mode": False,
            "num_ctx": int(os.getenv("SCENARIO_TAGGED_NUM_CTX", "12000")),
            "num_predict": int(os.getenv("SCENARIO_TAGGED_NUM_PREDICT", "1800")),
            "repeat_penalty": float(os.getenv("SCENARIO_TAGGED_REPEAT_PENALTY", "1.05")),
        },
        "chat_text": {
            "model": os.getenv("OLLAMA_MODEL_CHAT_TEXT", "gemma3:27b"),
            "json_mode": False,
            "num_ctx": int(os.getenv("CHAT_TEXT_NUM_CTX", "16384")),
            "num_predict": int(os.getenv("CHAT_TEXT_NUM_PREDICT", "1800")),
            "repeat_penalty": float(os.getenv("CHAT_TEXT_REPEAT_PENALTY", "1.05")),
        },
        "translation_text": {
            "model": os.getenv("OLLAMA_MODEL_TRANSLATION", "gemma4:26b"),
            "json_mode": False,
            "num_ctx": int(os.getenv("TRANSLATION_TEXT_NUM_CTX", "8192")),
            "num_predict": int(os.getenv("TRANSLATION_TEXT_NUM_PREDICT", "1800")),
            "repeat_penalty": float(os.getenv("TRANSLATION_TEXT_REPEAT_PENALTY", "1.00")),
        },

        # fallback
        "default": {
            "model": os.getenv("OLLAMA_MODEL", "gemma3:27b"),
            "json_mode": False,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "1600")),
            "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.10")),
        },
    }

    return profiles.get(task, profiles["default"])


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

    if provider == "openai":
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        return ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            api_key=api_key,
            streaming=streaming,
        ) | StrOutputParser()

    from langchain_ollama import OllamaLLM

    profile = _ollama_task_defaults(task)

    resolved_model = model or profile["model"]
    resolved_num_ctx = int(num_ctx) if num_ctx is not None else profile["num_ctx"]
    resolved_num_predict = int(num_predict) if num_predict is not None else profile["num_predict"]
    resolved_json_mode = profile["json_mode"] if json_mode is None else bool(json_mode)
    resolved_repeat_penalty = (
        float(repeat_penalty) if repeat_penalty is not None else profile["repeat_penalty"]
    )

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

    return OllamaLLM(**kwargs)


