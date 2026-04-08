import os
import re
import json
import logging
from pathlib import Path
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


def _sanitize_blueprint_candidate(candidate: str) -> str:
    candidate = (candidate or "").strip()

    # remove obvious template junk / markdown leftovers
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

    # common broken-token artifact from synth output, e.g. ],/\n    "climax_choices"
    candidate = re.sub(r',\s*/\s*(?=\")', ', ', candidate)
    candidate = re.sub(r'(?m)^\s*/\s*$', '', candidate)

    # trailing commas before closing braces/brackets
    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)

    # accidental control characters
    candidate = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", candidate)
    return candidate


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

    candidate = _sanitize_blueprint_candidate(cleaned[start:end + 1])
    logger.info("extract_blueprint_json(): candidate preview: %r", candidate[:4000])

    try:
        parsed = json.loads(candidate, strict=False)
    except json.JSONDecodeError as e:
        logger.warning("extract_blueprint_json(): JSON parse error after local repair: %s", e)
        logger.warning("extract_blueprint_json(): failed candidate body: %r", candidate[:4000])
        raise

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


def is_roll_verdict_message(message: str) -> bool:
    text = message or ""
    return (
        "VERDICT" in text
        and ("[SYSTEM MESSAGE" in text or "ROLL_VERDICT:" in text)
    )


def has_roll_verdict(message: str) -> bool:
    return is_roll_verdict_message(message)


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


def _safe_json_loads(value: str | None):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


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
