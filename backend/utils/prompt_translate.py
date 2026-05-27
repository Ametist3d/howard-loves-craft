import copy
import json
import logging
import os
import re

# pylint: disable=import-error
from utils.helpers import get_llm

logger = logging.getLogger("keeper_ai.prompt_translate")

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(UTILS_DIR)
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")

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
    "hr": "Croatian",
}

_CYRILLIC_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")


def normalize_language_code(lang: str | None) -> str:
    raw = (lang or "en").strip().lower()
    return raw if raw else "en"


def get_language_name(lang: str | None) -> str:
    code = normalize_language_code(lang)
    return LANGUAGE_LABELS.get(code, code)


def _unique_preserve_terms(items: list[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items or []:
        term = str(item or "").strip()
        if not term or len(term) < 3:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)

    out.sort(key=len, reverse=True)
    return out


def _extract_roll_suffix_terms(text: str) -> list[str]:
    terms: list[str] = []
    for m in re.finditer(r"→\s*Roll\s+[A-Za-z][A-Za-z ()/\-]*", str(text or "")):
        term = m.group(0).strip()
        if term:
            terms.append(term)
    return terms


def _protect_preserve_terms(
    text: str,
    preserve_terms: list[str] | None,
    mapping: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    mapping = mapping if mapping is not None else {}
    protected = str(text or "")

    for term in _unique_preserve_terms(preserve_terms):
        if term not in protected:
            continue
        token = f"__KEEPER_PRESERVE_{len(mapping)}__"
        mapping[token] = term
        protected = protected.replace(term, token)

    return protected, mapping


def _restore_preserve_terms(text: str, mapping: dict[str, str]) -> str:
    restored = str(text or "")
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    return restored


def _collect_preserve_terms_from_result(result: dict) -> list[str]:
    if not isinstance(result, dict):
        return []

    terms: list[str] = []

    def add(value):
        value = str(value or "").strip()
        if value:
            terms.append(value)

    updates = result.get("state_updates") or {}
    if isinstance(updates, dict):
        for key in ("character_name", "location_name"):
            add(updates.get(key, ""))

    rr = result.get("roll_request") or {}
    if isinstance(rr, dict):
        add(rr.get("skill_name", ""))

    combat_action = result.get("combat_action") or {}
    if isinstance(combat_action, dict):
        for key in ("actor_name", "target_name", "skill_name", "weapon_name"):
            add(combat_action.get(key, ""))

    scene_entities = result.get("scene_entities") or {}
    if isinstance(scene_entities, dict):
        for name in scene_entities.get("present_named_entities") or []:
            add(name)

    for action in result.get("suggested_actions") or []:
        terms.extend(_extract_roll_suffix_terms(str(action or "")))

    return _unique_preserve_terms(terms)


def _extract_first_json_object(raw: str) -> dict | None:
    text = str(raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1], strict=False)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def translate_player_action_to_english(text: str, source_language: str | None = None) -> str:
    """
    Canonicalize free-text player action into English for internal adjudication.

    Clicked suggested actions should bypass this through last_suggested_action_map_json.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    lang = normalize_language_code(source_language)
    if lang == "en":
        return raw

    prompt = (
        "Translate this tabletop RPG player action into concise English.\n"
        "Return only the translated action between the tags.\n\n"
        "STRICT RULES:\n"
        "- Preserve personal names exactly.\n"
        "- Preserve location names exactly when possible.\n"
        "- Preserve Call of Cthulhu skill names exactly if present.\n"
        "- Preserve object names when they look like proper nouns.\n"
        "- Preserve the force/intensity of the verb exactly: ask, question, interrogate, press, demand, threaten, sneak, inspect, decipher, analyze, etc.\n"
        "- Do not soften hostile or forceful actions.\n"
        "- Keep the action as an action, not an outcome.\n"
        "- Do not explain.\n\n"
        "<PLAYER_ACTION>\n"
        f"{raw}\n"
        "</PLAYER_ACTION>\n\n"
        "Output format:\n"
        "<ENGLISH_ACTION>...</ENGLISH_ACTION>"
    )

    try:
        translated = get_llm(
            temperature=0.0,
            task="translation_text",
            json_mode=False,
        ).invoke(prompt).strip()
        m = re.search(r"<ENGLISH_ACTION>\s*(.*?)\s*</ENGLISH_ACTION>", translated, re.DOTALL | re.IGNORECASE)
        if m:
            out = re.sub(r"\s+", " ", m.group(1).strip())
            return out or raw
        return raw
    except Exception as e:
        logger.warning("translate_player_action_to_english failed | lang=%s | err=%s", lang, e)
        return raw


def translate_chat_display_payload_for_user(result: dict, language: str) -> dict:
    """
    Translate the whole user-facing chat payload in one LLM call.

    Canonical DB state is English. This returned object is display-only.
    """
    safe = copy.deepcopy(result or {})
    language = normalize_language_code(language)
    if language == "en":
        return safe

    preserve_terms = _collect_preserve_terms_from_result(safe)
    target_language = get_language_name(language)

    payload = {
        "narrative": str(safe.get("narrative", "") or ""),
        "suggested_actions": [
            str(x or "").strip()
            for x in (safe.get("suggested_actions") or [])
            if str(x or "").strip()
        ],
        "state_updates": {
            "location_name": str((safe.get("state_updates") or {}).get("location_name", "") or ""),
            "location_description": str((safe.get("state_updates") or {}).get("location_description", "") or ""),
            "clue_found": str((safe.get("state_updates") or {}).get("clue_found", "") or ""),
            "clue_content": str((safe.get("state_updates") or {}).get("clue_content", "") or ""),
            "thread_progress": str((safe.get("state_updates") or {}).get("thread_progress", "") or ""),
        },
        "roll_request": {
            "action_text": str((safe.get("roll_request") or {}).get("action_text", "") or ""),
            "reason": str((safe.get("roll_request") or {}).get("reason", "") or ""),
        },
    }

    preserve_mapping: dict[str, str] = {}

    def protect(value: str) -> str:
        nonlocal preserve_mapping
        protected, preserve_mapping = _protect_preserve_terms(value, preserve_terms, preserve_mapping)
        return protected

    protected_payload = {
        "narrative": protect(payload["narrative"]),
        "suggested_actions": [protect(x) for x in payload["suggested_actions"]],
        "state_updates": {key: protect(value) for key, value in payload["state_updates"].items()},
        "roll_request": {key: protect(value) for key, value in payload["roll_request"].items()},
    }

    prompt = (
        f"Translate this JSON payload into {target_language} for player display.\n"
        "Return STRICT JSON only with the exact same keys and structure.\n\n"
        "STRICT RULES:\n"
        "- Translate natural prose into the target language.\n"
        "- Preserve every token like __KEEPER_PRESERVE_0__ exactly.\n"
        "- Preserve personal names, organization names, location labels, skill names, technical identifiers, and roll suffixes exactly.\n"
        "- Preserve substrings like `→ Roll Persuade` exactly.\n"
        "- Do not translate skill names after `→ Roll`.\n"
        "- Do not add keys.\n"
        "- Do not remove keys.\n"
        "- Do not add explanations.\n\n"
        "INPUT JSON:\n"
        f"{json.dumps(protected_payload, ensure_ascii=False)}"
    )

    try:
        raw = get_llm(
            temperature=0.05,
            task="translation_text",
            json_mode=False,
        ).invoke(prompt).strip()
        translated = _extract_first_json_object(raw)
        if not translated:
            logger.warning("translate_chat_display_payload_for_user: no JSON returned")
            return safe

        def restore(value: str) -> str:
            return _restore_preserve_terms(str(value or ""), preserve_mapping)

        safe["narrative"] = restore(translated.get("narrative", safe.get("narrative", "")))

        translated_actions = translated.get("suggested_actions", safe.get("suggested_actions", []))
        if isinstance(translated_actions, list) and len(translated_actions) == len(payload["suggested_actions"]):
            safe["suggested_actions"] = [restore(x) for x in translated_actions]

        updates = safe.get("state_updates") or {}
        translated_updates = translated.get("state_updates", {})
        if isinstance(updates, dict) and isinstance(translated_updates, dict):
            for key in ("location_name", "location_description", "clue_found", "clue_content", "thread_progress"):
                if key in updates:
                    updates[key] = restore(translated_updates.get(key, updates.get(key, "")))
            safe["state_updates"] = updates

        rr = safe.get("roll_request") or {}
        translated_rr = translated.get("roll_request", {})
        if isinstance(rr, dict) and isinstance(translated_rr, dict):
            if "action_text" in rr:
                rr["action_text"] = restore(translated_rr.get("action_text", rr.get("action_text", "")))
            if "reason" in rr:
                rr["reason"] = restore(translated_rr.get("reason", rr.get("reason", "")))
            safe["roll_request"] = rr

        return safe
    except Exception as e:
        logger.warning("translate_chat_display_payload_for_user failed for language=%s: %s", language, e)
        return safe


# Compatibility wrappers. Use sparingly; chat display should use translate_chat_display_payload_for_user().
def translate_text_for_user(text: str, language: str, preserve_terms: list[str] | None = None) -> str:
    language = normalize_language_code(language)
    text = str(text or "").strip()
    if not text or language == "en":
        return text

    preserve_terms = _unique_preserve_terms((preserve_terms or []) + _extract_roll_suffix_terms(text))
    protected_text, preserve_mapping = _protect_preserve_terms(text, preserve_terms)
    target_language = get_language_name(language)

    prompt = (
        f"Translate the source text into {target_language}.\n"
        "Return only the translation between the tags.\n\n"
        "STRICT RULES:\n"
        "- Preserve every token like __KEEPER_PRESERVE_0__ exactly.\n"
        "- Preserve personal names, organization names, location labels, skill names, technical identifiers, and roll suffixes exactly.\n"
        "- Do not translate, transliterate, decline, localize, or substitute names.\n\n"
        "<SOURCE_TEXT>\n"
        f"{protected_text}\n"
        "</SOURCE_TEXT>\n\n"
        "Output format:\n"
        "<TRANSLATION>...</TRANSLATION>"
    )

    try:
        translated = get_llm(temperature=0.1, task="translation_text", json_mode=False).invoke(prompt).strip()
        m = re.search(r"<TRANSLATION>\s*(.*?)\s*</TRANSLATION>", translated, re.DOTALL | re.IGNORECASE)
        if m:
            return _restore_preserve_terms(m.group(1).strip() or protected_text, preserve_mapping)
        return text
    except Exception as e:
        logger.warning("translate_text_for_user failed for language=%s: %s", language, e)
        return text


def translate_text_list_for_user(items: list[str], language: str, preserve_terms: list[str] | None = None) -> list[str]:
    cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]
    if normalize_language_code(language) == "en" or not cleaned:
        return cleaned
    fake = {"narrative": "", "suggested_actions": cleaned, "state_updates": {}, "roll_request": {}}
    translated = translate_chat_display_payload_for_user(fake, language)
    out = translated.get("suggested_actions") or cleaned
    return out if isinstance(out, list) and len(out) == len(cleaned) else cleaned


def translate_scenario_summary_for_user(summary_text: str, language: str) -> str:
    if os.getenv("TRANSLATE_SCENARIO_DISPLAY", "0").strip() != "1":
        return summary_text
    return translate_text_for_user(summary_text, language)


def translate_blueprint_for_display(blueprint: dict, language: str) -> dict:
    # Canonical blueprint stays English. Avoid hidden field-by-field translation cost by default.
    return copy.deepcopy(blueprint) if isinstance(blueprint, dict) else {}
