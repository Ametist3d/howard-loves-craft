import os
import re
import logging
from pathlib import Path
from typing import Iterable

from utils.helpers import get_llm

logger = logging.getLogger("keeper_ai.prompt_translate")

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))              # backend/utils
BACKEND_DIR = os.path.dirname(UTILS_DIR)                           # backend
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")                 # backend/prompts
DATA_DIR = os.path.join(BACKEND_DIR, "data")                       # backend/data
TRANSLATED_PROMPTS_ROOT = os.path.join(DATA_DIR, "sessions", "translated_prompts")

PROMPT_FILENAMES = (
    "character_gen.txt",
    "keeper_chat.txt",
    "scenario_gen.txt",
)

LANGUAGE_LABELS = {
    "ua": "Ukrainian",
    "en": "English",
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "pl": "Polish",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
}

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def _language_name(language: str) -> str:
    return LANGUAGE_LABELS.get((language or "en").lower(), language or "English")


def _session_dir(session_id: str, language: str) -> str:
    safe_session = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "global")
    safe_lang = re.sub(r"[^a-zA-Z0-9_.-]", "_", language or "en")
    return os.path.join(TRANSLATED_PROMPTS_ROOT, safe_session, safe_lang)


def _protect_template_vars(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        token = f"__TPLVAR_{len(mapping)}__"
        mapping[token] = match.group(0)
        return token

    return _PLACEHOLDER_RE.sub(repl, text), mapping


def _restore_template_vars(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def _translate_text(source_text: str, language: str) -> str:
    language = (language or "en").lower()
    if language == "en":
        return source_text

    protected_text, mapping = _protect_template_vars(source_text)
    llm = get_llm(temperature=0.1)
    target_language = _language_name(language)

    translation_prompt = (
        "You are translating an LLM system prompt for a game master application.\n"
        f"Translate the prompt into {target_language}.\n\n"
        "STRICT RULES:\n"
        "- Preserve every token like __TPLVAR_0__ exactly.\n"
        "- Preserve JSON keys exactly when they are inside double quotes.\n"
        "- Preserve dice notation, markdown structure, bullet structure, and enum values.\n"
        "- Preserve proper nouns unless translation is obvious and safe.\n"
        "- Do not add commentary. Output only the translated prompt text.\n\n"
        "PROMPT TO TRANSLATE:\n"
        f"{protected_text}"
    )

    translated = llm.invoke(translation_prompt).strip()
    translated = _restore_template_vars(translated, mapping)
    return translated if translated else source_text


def ensure_translated_prompts(session_id: str, language: str, filenames: Iterable[str] = PROMPT_FILENAMES) -> str:
    out_dir = _session_dir(session_id, language)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        src_path = os.path.join(PROMPTS_DIR, filename)
        dst_path = os.path.join(out_dir, filename)

        if os.path.exists(dst_path):
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        translated = _translate_text(source_text, language)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(translated)

        logger.info("Prepared translated prompt: %s [%s]", filename, language)

    return out_dir


def get_translated_prompt_path(session_id: str, language: str, filename: str) -> str:
    return os.path.join(_session_dir(session_id, language), filename)
