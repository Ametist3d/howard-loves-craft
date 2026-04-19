import os
import re
import json
import copy
import logging
from pathlib import Path
from typing import Iterable

#pylint: disable=import-error
from utils.helpers import get_llm

logger = logging.getLogger("keeper_ai.prompt_translate")

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/utils
BACKEND_DIR = os.path.dirname(UTILS_DIR)  # backend
PROMPTS_DIR = os.path.join(BACKEND_DIR, "prompts")  # backend/prompts
DATA_DIR = os.path.join(BACKEND_DIR, "data")  # backend/data
TRANSLATED_PROMPTS_ROOT = os.path.join(DATA_DIR, "sessions", "translated_prompts")

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
_XML_TAG_RE = re.compile(r"</?[A-Z0-9_]+>")
_DOUBLE_BRACE_RE = re.compile(r"\{\{|\}\}")

_CYRILLIC_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")

PROMPT_FILENAMES = (
    "character_gen.txt",
    "scenario_gen.txt",
    "scenario/scenario_plan.txt",
    "scenario/scenario_compose.txt",
    "scenario/scenario_instructions.txt",
    "keeper/header.txt",
    "keeper/core_identity.txt",
    "keeper/output_contract.txt",
    "keeper/action_adjudication.txt",
    "keeper/roll_resolution.txt",
    "keeper/scene_progression.txt",
    "keeper/opening_scene.txt",
)

ENGLISH_ONLY_PROMPTS = {
    "scenario_gen.txt",
    "scenario/scenario_plan.txt",
    "scenario/scenario_compose.txt",
    "scenario/scenario_instructions.txt",
    "keeper/header.txt",
    "keeper/core_identity.txt",
    "keeper/output_contract.txt",
    "keeper/action_adjudication.txt",
    "keeper/roll_resolution.txt",
    "keeper/scene_progression.txt",
    "keeper/opening_scene.txt",
}

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

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def get_language_name(lang: str | None) -> str:
    code = normalize_language_code(lang)
    return LANGUAGE_LABELS.get(code, code)

def normalize_language_code(lang: str | None) -> str:
    raw = (lang or "en").strip().lower()
    return raw if raw else "en"

def _language_name(language: str) -> str:
    return LANGUAGE_LABELS.get((language or "en").lower(), language or "English")


def _session_dir(session_id: str, language: str) -> str:
    safe_session = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "global")
    safe_lang = re.sub(r"[^a-zA-Z0-9_.-]", "_", language or "en")
    return os.path.join(TRANSLATED_PROMPTS_ROOT, safe_session, safe_lang)


def _protect_special_tokens(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}

    def store(token_text: str) -> str:
        token = f"__KEEPER_TOKEN_{len(mapping)}__"
        mapping[token] = token_text
        return token

    # protect template vars first
    text = _PLACEHOLDER_RE.sub(lambda m: store(m.group(0)), text)

    # protect machine tags like <SYSTEM_RESPONSE_JSON>
    text = _XML_TAG_RE.sub(lambda m: store(m.group(0)), text)

    # protect doubled braces used for PromptTemplate escaping
    text = _DOUBLE_BRACE_RE.sub(lambda m: store(m.group(0)), text)

    return text, mapping


def _restore_special_tokens(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text

def _translate_text(source_text: str, language: str) -> str:
    language = (language or "en").lower()
    if language == "en":
        return source_text

    protected_text, mapping = _protect_special_tokens(source_text)
    llm = get_llm(
        temperature=0.1,
        task="translation_text",
    )
    target_language = _language_name(language)

    translation_prompt = (
        "You are translating an LLM system prompt for a game master application.\n"
        f"Translate the prompt into {target_language}.\n\n"
        "STRICT RULES:\n"
        "- Preserve every token like __KEEPER_TOKEN_0__ exactly.\n"
        "- Preserve JSON keys exactly when they are inside double quotes.\n"
        "- Preserve dice notation, markdown structure, bullet structure, and enum values.\n"
        "- Preserve proper nouns unless translation is obvious and safe.\n"
        "- Do not add commentary. Output only the translated prompt text.\n\n"
        "PROMPT TO TRANSLATE:\n"
        f"{protected_text}"
    )

    translated = llm.invoke(translation_prompt).strip()
    translated = _restore_special_tokens(translated, mapping)
    return translated if translated else source_text


def ensure_translated_prompts(session_id: str, language: str, filenames: Iterable[str] = PROMPT_FILENAMES) -> str:
    out_dir = _session_dir(session_id, language)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        src_path = os.path.join(PROMPTS_DIR, filename)
        dst_path = os.path.join(out_dir, filename)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source prompt not found: {src_path}")

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(dst_path):
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        if filename in ENGLISH_ONLY_PROMPTS:
            translated = source_text
        else:
            translated = _translate_text(source_text, language)

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(translated)

        logger.info("Prepared translated prompt: %s [%s]", filename, language)

    return out_dir

def get_translated_prompt_path(session_id: str, language: str, filename: str) -> str:
    return os.path.join(_session_dir(session_id, language), filename)


def _looks_translated(text: str, language: str) -> bool:
    language = normalize_language_code(language)
    text = str(text or "").strip()

    if not text:
        return False
    if language == "en":
        return True
    if language == "ua":
        return bool(_CYRILLIC_RE.search(text))

    # for other languages keep simple fallback
    return True


def translate_text_for_user(text: str, language: str) -> str:
    language = normalize_language_code(language)
    text = str(text or "").strip()

    if not text or language == "en":
        return text

    target_language = get_language_name(language)

    prompt = (
        f"Translate the source text into {target_language}.\n"
        "Return only the translation between the tags.\n\n"
        "<SOURCE_TEXT>\n"
        f"{text}\n"
        "</SOURCE_TEXT>\n\n"
        "Output format:\n"
        "<TRANSLATION>...</TRANSLATION>"
    )

    try:
        translated = get_llm(
            temperature=0.1,
            task="translation_text",
            json_mode=False,
        ).invoke(prompt).strip()

        m = re.search(r"<TRANSLATION>\s*(.*?)\s*</TRANSLATION>", translated, re.DOTALL)
        if m:
            return m.group(1).strip() or text

        return text
    except Exception as e:
        logger.warning("translate_text_for_user failed for language=%s: %s", language, e)
        return text


def translate_text_list_for_user(items: list[str], language: str) -> list[str]:
    language = normalize_language_code(language)
    cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]

    if language == "en" or not cleaned:
        return cleaned

    sep = "\n<<<ITEM>>>\n"
    joined = sep.join(cleaned)
    target_language = get_language_name(language)

    prompt = (
        f"Translate each item below into {target_language}.\n"
        "STRICT RULES:\n"
        "- Output only the translated items.\n"
        f"- Keep the separator exactly as: {sep.strip()}\n"
        "- Translate each item fully into the target language.\n"
        "- Preserve proper nouns only.\n"
        "- Do not add commentary.\n\n"
        "ITEMS:\n"
        f"{joined}"
    )

    try:
        translated = get_llm(
            temperature=0.1,
            task="translation_text",
            json_mode=False,
        ).invoke(prompt).strip()

        if not translated:
            return cleaned

        parts = [p.strip() for p in translated.split(sep)]
        if len(parts) != len(cleaned):
            return cleaned

        if all(_looks_translated(p, language) for p in parts):
            return parts

        retry_prompt = (
            f"Rewrite each item fully in {target_language}.\n"
            "STRICT RULES:\n"
            "- Output only the translated items.\n"
            f"- Keep the separator exactly as: {sep.strip()}\n"
            "- Every item must be in the target language.\n"
            "- Preserve proper nouns only.\n\n"
            "ITEMS:\n"
            f"{joined}"
        )

        translated_retry = get_llm(
            temperature=0.05,
            task="translation_text",
            json_mode=False,
        ).invoke(retry_prompt).strip()

        retry_parts = [p.strip() for p in translated_retry.split(sep)]
        if len(retry_parts) == len(cleaned) and all(_looks_translated(p, language) for p in retry_parts):
            return retry_parts

        logger.warning("translate_text_list_for_user: translation did not switch language | lang=%s", language)
        return cleaned

    except Exception as e:
        logger.warning("translate_text_list_for_user failed for language=%s: %s", language, e)
        return cleaned


def translate_opening_result_for_user(result: dict, language: str) -> dict:
    """
    Translate only user-facing opening content.
    Keep state_updates / roll_request canonical for internal logic.
    """
    safe = dict(result or {})
    language = normalize_language_code(language)

    if language == "en":
        return safe

    safe["narrative"] = translate_text_for_user(safe.get("narrative", ""), language)
    safe["suggested_actions"] = translate_text_list_for_user(
        safe.get("suggested_actions", []),
        language,
    )
    return safe


def translate_scenario_summary_for_user(summary_text: str, language: str) -> str:
    return translate_text_for_user(summary_text, language)

def translate_blueprint_for_display(blueprint: dict, language: str) -> dict:
    """
    Optional translated copy for UI/debug viewing.
    Canonical blueprint in DB should stay English.
    Translate field-by-field to avoid whole-JSON corruption on local models.
    """
    language = normalize_language_code(language)
    if language == "en" or not isinstance(blueprint, dict):
        return blueprint if isinstance(blueprint, dict) else {}

    out = copy.deepcopy(blueprint)

    top_level_fields = [
        "title",
        "era_and_setting",
        "atmosphere_notes",
        "inciting_hook",
        "core_mystery",
        "hidden_threat",
        "truth_the_players_never_suspect",
    ]
    for key in top_level_fields:
        if key in out:
            out[key] = translate_text_for_user(out.get(key, ""), language)

    engine = out.get("scenario_engine") or {}
    for key in ("surface_explanation", "actual_explanation"):
        if key in engine:
            engine[key] = translate_text_for_user(engine.get(key, ""), language)

    for key in ("false_leads", "contradictions", "reversals", "dynamic_pressures"):
        engine[key] = translate_text_list_for_user(engine.get(key, []), language)

    translated_choices = []
    for choice in engine.get("climax_choices", []) or []:
        translated_choices.append({
            "option": translate_text_for_user(choice.get("option", ""), language),
            "cost": translate_text_for_user(choice.get("cost", ""), language),
            "consequence": translate_text_for_user(choice.get("consequence", ""), language),
        })
    engine["climax_choices"] = translated_choices
    out["scenario_engine"] = engine

    translated_acts = []
    for act in out.get("acts", []) or []:
        act_copy = dict(act)
        for key in ("title", "summary", "purpose", "belief_shift"):
            act_copy[key] = translate_text_for_user(act_copy.get(key, ""), language)

        act_copy["required_payoffs"] = translate_text_list_for_user(
            act_copy.get("required_payoffs", []),
            language,
        )

        translated_scenes = []
        for scene in act_copy.get("scenes", []) or []:
            scene_copy = dict(scene)

            for key in (
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
                "keeper_notes",
            ):
                scene_copy[key] = translate_text_for_user(scene_copy.get(key, ""), language)

            for list_key in ("reveals", "conceals", "clues_available", "npc_present"):
                scene_copy[list_key] = translate_text_list_for_user(
                    scene_copy.get(list_key, []),
                    language,
                )

            translated_scenes.append(scene_copy)

        act_copy["scenes"] = translated_scenes
        translated_acts.append(act_copy)

    out["acts"] = translated_acts

    translated_locations = []
    for loc in out.get("locations", []) or []:
        loc_copy = dict(loc)
        for key in ("name", "description", "hidden"):
            loc_copy[key] = translate_text_for_user(loc_copy.get(key, ""), language)
        if "tags" in loc_copy:
            if isinstance(loc_copy["tags"], list):
                loc_copy["tags"] = translate_text_list_for_user(loc_copy["tags"], language)
            else:
                loc_copy["tags"] = translate_text_for_user(str(loc_copy["tags"]), language)
        translated_locations.append(loc_copy)
    out["locations"] = translated_locations

    translated_npcs = []
    for npc in out.get("npcs", []) or []:
        npc_copy = dict(npc)
        for key in ("name", "description", "role", "secret", "motivation"):
            npc_copy[key] = translate_text_for_user(npc_copy.get(key, ""), language)
        translated_npcs.append(npc_copy)
    out["npcs"] = translated_npcs

    translated_clues = []
    for clue in out.get("clues", []) or []:
        clue_copy = dict(clue)
        for key in ("title", "content", "true_meaning", "location"):
            clue_copy[key] = translate_text_for_user(clue_copy.get(key, ""), language)
        translated_clues.append(clue_copy)
    out["clues"] = translated_clues

    translated_threads = []
    for thread in out.get("plot_threads", []) or []:
        thread_copy = dict(thread)
        for key in ("name", "stakes"):
            thread_copy[key] = translate_text_for_user(thread_copy.get(key, ""), language)
        translated_threads.append(thread_copy)
    out["plot_threads"] = translated_threads

    return out

def translate_keeper_result_for_user(result: dict, language: str) -> dict:
    """
    Translate only user-facing content for chat display.
    Keep canonical internal fields (skill_name, combat payload, roll_request.skill_name) unchanged.
    """
    safe = copy.deepcopy(result or {})
    language = normalize_language_code(language)

    if language == "en":
        return safe

    safe["narrative"] = translate_text_for_user(safe.get("narrative", ""), language)
    safe["suggested_actions"] = translate_text_list_for_user(
        safe.get("suggested_actions", []),
        language,
    )

    updates = safe.get("state_updates") or {}
    if isinstance(updates, dict):
        # translate only human-visible text fields
        for key in ("location_name", "location_description", "clue_found", "clue_content", "thread_progress"):
            if key in updates:
                updates[key] = translate_text_for_user(updates.get(key, ""), language)
        safe["state_updates"] = updates

    rr = safe.get("roll_request") or {}
    if isinstance(rr, dict):
        # keep skill_name canonical English
        if "action_text" in rr:
            rr["action_text"] = translate_text_for_user(rr.get("action_text", ""), language)
        if "reason" in rr:
            rr["reason"] = translate_text_for_user(rr.get("reason", ""), language)
        safe["roll_request"] = rr

    cu = safe.get("combat_update") or {}
    if isinstance(cu, dict):
        # keep skill_name / attack_mode / defender_option canonical
        for key in ("attacker_name", "target_name", "weapon_name"):
            if key in cu:
                cu[key] = translate_text_for_user(cu.get(key, ""), language)
        safe["combat_update"] = cu

    return safe


# # compatibility alias if you want to keep old call sites working
# def translate_opening_result_for_user(result: dict, language: str) -> dict:
#     return translate_keeper_result_for_user(result, language)