import asyncio
import json
import os
import re
import uuid
from pathlib import Path

from langchain_core.prompts import PromptTemplate

# pylint: disable=import-error
from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import ChatRequest, validate_chat_response_payload
from utils.helpers import (
    build_verdict_guard,
    extract_json,
    extract_last_turn_ban,
    get_chat_history,
    get_llm,
    has_roll_verdict,
    parse_roll_resolution_from_message,
)
from utils.prompt_translate import normalize_language_code, translate_chat_display_payload_for_user
from utils.helper_actions import (
    clear_pending_roll,
    intercept_player_action_for_roll_gate,
    save_pending_roll,
    _detect_roll_request_from_suggested_actions,
    _resolve_player_action_to_canonical_english,
)
from utils.helper_story import (
    build_scene_loop_guard,
    build_stall_forcing_guard,
    build_state_continuity_guard,
    compress_story,
    detect_generated_loop,
    suppress_known_destination_suggestions,
)
from utils.helper_state import (
    apply_state_updates,
    assemble_keeper_prompt,
    build_authoritative_context,
    looks_like_valid_keeper_response,
    maybe_force_movement_progress,
    sanitize_llm_result_on_validation_failure,
    validate_llm_response_against_state,
)
from utils.combat import get_combat_state, resolve_combat_turn
from utils.engine import (
    SESSIONS_DIR,
    _image_results,
    _is_combat_turn,
    _should_refresh_digest,
    active_dbs,
    logger,
    rules_db,
)
from utils.rules_retrieval_patch import retrieve_with_rerank
from utils.rules_retrieval_config import RULES_RETRIEVAL_CFG
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL


_REQUEST_BODY_PATH = Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in _FALSE_ENV_VALUES


def _clone_request_with_message(req: ChatRequest, message: str) -> ChatRequest:
    copier = getattr(req, "model_copy", None)
    if callable(copier):
        return copier(update={"message": message})
    return req.copy(update={"message": message})


def _merge_setting_and_era(setting_override: str, era_override: str) -> str:
    setting = str(setting_override or "").strip()
    era = str(era_override or "").strip()

    if not setting:
        return era
    if not era:
        return setting
    if setting == era:
        return setting

    if " | " in setting and " | " not in era:
        return era
    if " | " in era and " | " not in setting:
        return setting

    return f"{setting} {era}"


def _ensure_session_db(session_id: str) -> SessionDB:
    if session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[session_id] = SessionDB(info.db_path)
    return active_dbs[session_id]


def _get_session_language_from_db(db: SessionDB) -> str:
    cur = db.conn.cursor()
    row = cur.execute("SELECT value FROM kv_store WHERE key='language'").fetchone()
    return normalize_language_code(row["value"] if row else "en")


def _kv_get(cur, key: str, default: str = "") -> str:
    row = cur.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
    return str(row["value"] if row else default)


def _get_visual_context(session_id: str) -> tuple[str, str]:
    visual_history = ""
    char_visuals = ""

    if session_id in active_dbs:
        db = active_dbs[session_id]
        cur = db.conn.cursor()

        row = cur.execute("SELECT value FROM kv_store WHERE key='visual_history'").fetchone()
        visual_history = row["value"] if row else ""

        rows = cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'").fetchall()
        if rows:
            char_visuals = "ESTABLISHED CHARACTERS (maintain consistent appearance):\n"
            char_visuals += "\n".join(f"- {r['value']}" for r in rows)

    return visual_history, char_visuals


def _compact_for_image(text: str, max_chars: int = 900) -> str:
    clean = re.sub(r"<[^>]+>", " ", str(text or ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_chars].rstrip()


def _build_deterministic_scene_prompt(
    *,
    narrative: str,
    era: str,
    setting: str,
    visual_history: str = "",
    char_visuals: str = "",
) -> str:
    """
    Build an image prompt without another LLM call.

    Previous build_scene_prompt() used get_llm(), which added one GPT/Ollama call per turn.
    This deterministic prompt keeps image generation available without hidden chat-completions cost.
    """
    parts = [
        "Painterly digital illustration of the current Call of Cthulhu scene.",
    ]

    merged_context = _compact_for_image(_merge_setting_and_era(setting, era), 260)
    if merged_context:
        parts.append(f"Era and setting: {merged_context}.")

    scene = _compact_for_image(narrative, 700)
    if scene:
        parts.append(f"Scene focus: {scene}.")

    visuals = _compact_for_image(char_visuals, 250)
    if visuals:
        parts.append(f"Maintain established character visuals: {visuals}.")

    history = _compact_for_image(visual_history, 220)
    if history:
        parts.append(f"Maintain visual continuity with previous scenes: {history}.")

    parts.append("Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.")
    return " ".join(parts)


async def _generate_image_bg(
    *,
    generation_id: str,
    session_id: str,
    narrative: str,
    setting: str,
    era: str,
) -> None:
    try:
        from img_gen.comfy_client import ComfyClient

        comfy = ComfyClient(_COMFY_BASE_URL)
        visual_history, char_visuals = _get_visual_context(session_id)

        scene_prompt = _build_deterministic_scene_prompt(
            narrative=narrative,
            era=era,
            setting=setting,
            visual_history=visual_history,
            char_visuals=char_visuals,
        )
        logger.info("Image prompt [%s]: %s", generation_id, scene_prompt)

        with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as f:
            body = json.load(f)

        body["params"]["prompt"] = scene_prompt

        img_result = comfy.generate(body)
        _image_results[generation_id] = img_result["image_url"]
        logger.info("Image ready [%s]: %s", generation_id, img_result["image_url"])

    except Exception as e:
        logger.warning("Image generation failed [%s]: %s", generation_id, e)
        _image_results[generation_id] = None


def _attach_image_generation(
    *,
    session_id: str,
    display_result: dict,
    canonical_narrative: str,
    setting: str,
    era: str,
) -> dict:
    if not _env_enabled("KEEPER_ENABLE_SCENE_IMAGES", "1"):
        display_result["generation_id"] = None
        display_result["image_url"] = None
        return display_result

    gen_id = uuid.uuid4().hex
    _image_results[gen_id] = "pending"

    display_result["generation_id"] = gen_id
    display_result["image_url"] = None

    asyncio.create_task(
        _generate_image_bg(
            generation_id=gen_id,
            session_id=session_id,
            narrative=canonical_narrative,
            setting=setting,
            era=era,
        )
    )
    return display_result


def _store_display_to_canonical_action_map(db: SessionDB, canonical_result: dict, display_result: dict) -> None:
    canonical_actions = [
        str(x).strip()
        for x in (canonical_result.get("suggested_actions") or [])
        if str(x).strip()
    ]
    display_actions = [
        str(x).strip()
        for x in (display_result.get("suggested_actions") or [])
        if str(x).strip()
    ]

    action_map = {}
    for shown, canonical in zip(display_actions, canonical_actions):
        if shown and canonical:
            action_map[shown] = canonical

    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('last_suggested_action_map_json', ?)",
        (json.dumps(action_map, ensure_ascii=False),),
    )
    db.conn.commit()


def _normalize_for_schema(result: dict) -> dict:
    """
    Normalize public ChatResponse schema fields, but preserve internal runtime flags.

    Important:
    validate_chat_response_payload() / Pydantic may drop unknown keys such as:
    - _player_movement_intent
    - _scene_advance_allowed
    - _movement_from_story_scene_id

    Those flags are needed later by apply_state_updates() -> advance_story_progress().
    """
    incoming = dict(result or {})
    private_flags = {
        k: v
        for k, v in incoming.items()
        if isinstance(k, str) and k.startswith("_")
    }

    safe = dict(incoming)

    if not isinstance(safe.get("state_updates"), dict):
        safe["state_updates"] = {}
    if not isinstance(safe.get("combat_action"), dict):
        safe["combat_action"] = {}
    if not isinstance(safe.get("roll_request"), dict):
        safe["roll_request"] = {}
    if not isinstance(safe.get("scene_entities"), dict):
        safe["scene_entities"] = {"present_named_entities": []}
    if not isinstance(safe.get("suggested_actions"), list):
        safe["suggested_actions"] = []

    normalized = validate_chat_response_payload(safe)
    normalized.update(private_flags)
    return normalized


def filter_scene_entities_against_db(db: SessionDB, result: dict) -> dict:
    known = {
        str(a.get("name", "") or "").strip()
        for a in db.list_actors()
        if str(a.get("name", "") or "").strip()
    }

    entities = result.get("scene_entities") or {}
    names = entities.get("present_named_entities") or []

    filtered = [str(n).strip() for n in names if str(n).strip() in known]
    unknown = [str(n).strip() for n in names if str(n).strip() and str(n).strip() not in known]

    if unknown:
        logger.warning("DROPPED_UNKNOWN_SCENE_ENTITIES: %s", unknown)

    result["scene_entities"] = {"present_named_entities": filtered}
    return result


def _append_rules_context(*, context_str: str, rule_query: str, top_k: int) -> str:
    """
    Live chat retrieves rules/mechanics only.
    Scenario fiction comes from the session DB story graph, not global scen_db.
    """
    if not rules_db:
        return context_str

    try:
        rule_hits = retrieve_with_rerank(rules_db, rule_query, cfg=RULES_RETRIEVAL_CFG)
        rule_docs = [hit.doc for hit in rule_hits]
    except Exception as e:
        logger.warning("rules retrieve_with_rerank failed; falling back to similarity_search: %s", e)
        rule_docs = rules_db.similarity_search(rule_query, k=top_k)

    rule_texts = [str(doc.page_content or "").strip() for doc in rule_docs if str(doc.page_content or "").strip()]
    if not rule_texts:
        return context_str

    return context_str + "\n\n--- RELEVANT RULEBOOK LORE ---\n" + "\n\n".join(rule_texts)


def _canonical_action_for_keeper(db: SessionDB, message: str) -> str:
    """
    Keeper runtime is canonical English.
    Display language is handled only after result parsing/finalization.
    """
    raw = str(message or "").strip()
    if not raw:
        return ""
    if has_roll_verdict(raw) or raw.startswith("[SYSTEM"):
        return raw
    return _resolve_player_action_to_canonical_english(db, raw)


def _is_start_message(message: str) -> bool:
    lower = str(message or "").lower()
    return "start the story" in lower or "open with the first playable scene" in lower


def _is_system_message(message: str) -> bool:
    return str(message or "").strip().startswith("[SYSTEM")


def _load_turn_context(db: SessionDB, req: ChatRequest, canonical_action: str) -> dict:
    cur = db.conn.cursor()

    campaign_atoms = _kv_get(cur, "scenario_atoms", "")
    themes = _kv_get(cur, "scenario_themes", "STANDARD")
    setting_override = _kv_get(cur, "scenario_setting", "Lovecraftian Horror Lore")
    era_override = _kv_get(cur, "era_context", "1920s Lovecraftian Horror")
    session_language = _get_session_language_from_db(db)
    prompt_dir = _kv_get(cur, "prompt_dir", "") or None

    context_str, state_str = build_authoritative_context(
        db,
        campaign_atoms=campaign_atoms,
        themes=themes,
    )

    is_start_msg = _is_start_message(req.message)
    combat_turn = _is_combat_turn(db, canonical_action)

    if req.rag_enabled and not is_start_msg:
        rule_query = canonical_action
        if combat_turn:
            rule_query = (
                f"{canonical_action}\n"
                "Call of Cthulhu 7e combat sequence DEX order melee fight back dodge "
                "firearms dive for cover readied firearm point blank outnumbered "
                "major wound unconscious dying instant death"
            )
        context_str = _append_rules_context(context_str=context_str, rule_query=rule_query, top_k=req.top_k)

    verdict_guard = build_verdict_guard(req.message)
    scene_loop_guard = build_scene_loop_guard(db)
    stall_guard = build_stall_forcing_guard(db)
    continuity_guard = build_state_continuity_guard(db)

    combat_state = get_combat_state(db) if combat_turn else {}
    combat_state_text = json.dumps(combat_state, ensure_ascii=False)

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=has_roll_verdict(req.message),
        include_scene_progression=True,
        include_opening_scene=is_start_msg,
        prompt_dir=prompt_dir,
    )

    campaign_context = (
        context_str
        + "\n\n--- CURRENT GAME STATE ---\n"
        + state_str
        + "\n\n--- CURRENT COMBAT STATE ---\n"
        + combat_state_text
        + "\n\n"
        + scene_loop_guard
        + ("\n\n" + continuity_guard if continuity_guard else "")
        + ("\n\n" + stall_guard if stall_guard else "")
        + ("\n\n" + verdict_guard if verdict_guard else "")
    )

    prompt_vars = {
        # Internal Keeper language is always English.
        "language": "en",
        "language_name": "English",
        "campaign_context": campaign_context,
        "era_context": _merge_setting_and_era(setting_override, era_override),
        "history": get_chat_history(db, limit=15),
        "action": canonical_action,
        "last_turn_ban": extract_last_turn_ban(db),
    }

    return {
        "cur": cur,
        "session_language": session_language,
        "setting_override": setting_override,
        "era_override": era_override,
        "is_start_msg": is_start_msg,
        "combat_turn": combat_turn,
        "keeper_system_prompt": keeper_system_prompt,
        "prompt_vars": prompt_vars,
        "repair_prompt_vars_base": {
            **prompt_vars,
            "campaign_context_base": campaign_context,
        },
    }


def _build_repair_prompt_vars(turn: dict, repair_notice: str) -> dict:
    base_context = turn["repair_prompt_vars_base"]["campaign_context_base"]
    prompt_vars = dict(turn["prompt_vars"])
    prompt_vars["campaign_context"] = (
        base_context
        + "\n\n--- CONTRACT REPAIR NOTICE ---\n"
        + repair_notice
    )
    return prompt_vars

def _parse_and_postprocess_result(
    *,
    db: SessionDB,
    req: ChatRequest,
    canonical_action: str,
    raw_text: str,
    chain,
    turn: dict,
) -> dict:
    result = extract_json(raw_text)
    result = maybe_force_movement_progress(db, canonical_action, result)

    parsed_roll_resolution = parse_roll_resolution_from_message(req.message)
    if parsed_roll_resolution:
        result["roll_resolution"] = parsed_roll_resolution

    # ------------------------------------------------------------------
    # 1) Contract repair: only if the model failed SYSTEM_RESPONSE_JSON.
    # ------------------------------------------------------------------
    if not looks_like_valid_keeper_response(str(raw_text), result):
        logger.warning("Non-contract LLM reply detected; attempting one repair regeneration")

        repair_notice = (
            "Your previous reply violated the required Keeper output contract.\n"
            "Return ONLY a valid <SYSTEM_RESPONSE_JSON>...</SYSTEM_RESPONSE_JSON> block.\n"
            "Do not ask the user for clarification.\n"
            "Do not explain your answer.\n"
            "Do not output markdown.\n"
            "Preserve the same current scene, current action, and current game state.\n"
            "Internal language must be English.\n"
        )

        repaired_text = chain.invoke(_build_repair_prompt_vars(turn, repair_notice))
        logger.warning("Repair regeneration raw preview: %r", str(repaired_text)[:2000])

        repaired_result = extract_json(repaired_text)

        if looks_like_valid_keeper_response(str(repaired_text), repaired_result):
            result = maybe_force_movement_progress(db, canonical_action, repaired_result)

            # Preserve authoritative roll resolution if this turn is resolving a roll.
            if parsed_roll_resolution:
                result["roll_resolution"] = parsed_roll_resolution

    # ------------------------------------------------------------------
    # 2) Combat resolver remains authoritative when this is a combat turn.
    # ------------------------------------------------------------------
    if turn.get("combat_turn"):
        result = resolve_combat_turn(db, result)

    # ------------------------------------------------------------------
    # 3) Normalize public schema, then restore movement binding.
    #    Schema validation can drop internal _private flags.
    # ------------------------------------------------------------------
    result = _normalize_for_schema(result)

    if parsed_roll_resolution:
        result["roll_resolution"] = parsed_roll_resolution

    result = maybe_force_movement_progress(db, canonical_action, result)

    # ------------------------------------------------------------------
    # 4) State validation / sanitization.
    # ------------------------------------------------------------------
    violations = validate_llm_response_against_state(db, result)
    if violations:
        logger.warning("LLM response validation failed: %s", violations)

        db.log_event(
            "LLM_VALIDATION_FAIL",
            {
                "violations": violations,
                "narrative": result.get("narrative", "")[:500],
            },
        )

        result = sanitize_llm_result_on_validation_failure(db, result, violations)
        result = _normalize_for_schema(result)

        if parsed_roll_resolution:
            result["roll_resolution"] = parsed_roll_resolution

        # Reapply after sanitizer too, because sanitizer may rewrite state_updates.
        result = maybe_force_movement_progress(db, canonical_action, result)

    # ------------------------------------------------------------------
    # 5) Deterministic anti-loop check.
    #    This catches repeated clues / repeated destinations / soft loops
    #    that pass normal schema and state validation.
    # ------------------------------------------------------------------
    loop_violations = detect_generated_loop(db, result)
    if loop_violations:
        logger.warning("Generated loop detected: %s", loop_violations)

        db.log_event(
            "LLM_LOOP_DETECTED",
            {
                "violations": loop_violations,
                "narrative": result.get("narrative", "")[:500],
                "suggested_actions": result.get("suggested_actions", [])[:3],
                "state_updates": result.get("state_updates", {}),
            },
        )

        repair_notice = (
            "Your previous response repeated already-established progress or suggested an already-used lead.\n"
            f"Loop violations: {loop_violations}\n\n"
            "Repair the response now.\n"
            "Rules:\n"
            "- Do not rediscover existing clues.\n"
            "- Do not suggest following a map/lead to a location already reached.\n"
            "- Do not offer another inspect/search/consult step unless it reveals a new fact immediately.\n"
            "- The repaired response must materially change the situation through a new clue, threat, cost, choice, or location transition.\n"
            "- Keep the same current player action and current DB scene.\n"
            "- Return only valid <SYSTEM_RESPONSE_JSON>...</SYSTEM_RESPONSE_JSON>."
        )

        repaired_text = chain.invoke(_build_repair_prompt_vars(turn, repair_notice))
        logger.warning("Loop repair regeneration raw preview: %r", str(repaired_text)[:2000])

        repaired_result = extract_json(repaired_text)

        if looks_like_valid_keeper_response(str(repaired_text), repaired_result):
            result = maybe_force_movement_progress(db, canonical_action, repaired_result)
            result = _normalize_for_schema(result)

            if parsed_roll_resolution:
                result["roll_resolution"] = parsed_roll_resolution

            result = maybe_force_movement_progress(db, canonical_action, result)

            # One final normal state validation after loop repair.
            repair_violations = validate_llm_response_against_state(db, result)
            if repair_violations:
                logger.warning("Loop repair still violates state: %s", repair_violations)

                db.log_event(
                    "LLM_LOOP_REPAIR_VALIDATION_FAIL",
                    {
                        "violations": repair_violations,
                        "narrative": result.get("narrative", "")[:500],
                    },
                )

                result = sanitize_llm_result_on_validation_failure(db, result, repair_violations)
                result = _normalize_for_schema(result)

                if parsed_roll_resolution:
                    result["roll_resolution"] = parsed_roll_resolution

                result = maybe_force_movement_progress(db, canonical_action, result)

    # ------------------------------------------------------------------
    # 6) Final safety cleanup for suggestions that send players back to
    #    already reached destinations.
    # ------------------------------------------------------------------
    result = suppress_known_destination_suggestions(db, result)
    result = _normalize_for_schema(result)

    if parsed_roll_resolution:
        result["roll_resolution"] = parsed_roll_resolution

    result = maybe_force_movement_progress(db, canonical_action, result)

    return result


def _finalize_keeper_result(
    *,
    db: SessionDB,
    req: ChatRequest,
    result: dict,
    session_language: str,
    setting_override: str,
    era_override: str,
    cur,
) -> dict:
    """
    Single canonical post-processing path for both /api/chat and /api/chat/stream.

    Canonical result is English. Only display_result is localized.
    """
    result = _normalize_for_schema(result)
    result = filter_scene_entities_against_db(db, result)

    derived_rr = _detect_roll_request_from_suggested_actions(result.get("suggested_actions", []))
    if derived_rr and not (result.get("roll_request") or {}).get("required"):
        logger.info("ROLL_REQUEST_DERIVED_FROM_SUGGESTED_ACTIONS: %s", derived_rr)
        result["roll_request"] = derived_rr

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    # Canonical DB state first, display localization second.
    apply_state_updates(db, result)

    current_scene = db.get_current_story_scene()
    pcs = db.list_actors("PC")
    logger.info(
        "POST_APPLY_STATE current_story_scene=%r pc_locations=%r result_location=%r",
        (current_scene or {}).get("location_name", ""),
        [
            {
                "pc": pc.get("name"),
                "location_id": pc.get("location_id"),
            }
            for pc in pcs
        ],
        (result.get("state_updates") or {}).get("location_name", ""),
    )

    display_result = translate_chat_display_payload_for_user(result, session_language)
    _store_display_to_canonical_action_map(db, result, display_result)

    display_result = _attach_image_generation(
        session_id=req.session_id,
        display_result=display_result,
        canonical_narrative=result.get("narrative", ""),
        setting=setting_override,
        era=era_override,
    )

    db.log_event(
        "CHAT",
        {
            "role": "Keeper",
            "content": result.get("narrative", ""),
            "display_content": display_result.get("narrative", ""),
        },
    )

    row_tc = cur.execute("SELECT value FROM kv_store WHERE key='turn_count'").fetchone()
    turn_count = int(row_tc["value"]) + 1 if row_tc else 1
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)",
        (str(turn_count),),
    )
    db.conn.commit()

    if _should_refresh_digest(turn_count):
        asyncio.create_task(compress_story(db))
        logger.info("Story compression triggered at turn %s", turn_count)

    return display_result


async def _maybe_intercept_roll_gate(db: SessionDB, req: ChatRequest) -> dict | None:
    if has_roll_verdict(req.message):
        clear_pending_roll(db)
        return None

    if _is_start_message(req.message) or _is_system_message(req.message):
        return None

    return await intercept_player_action_for_roll_gate(db, req.message)


def _finalize_intercepted_result(db: SessionDB, req: ChatRequest, intercepted: dict) -> dict:
    session_language = _get_session_language_from_db(db)
    canonical = _normalize_for_schema(intercepted)
    display_result = translate_chat_display_payload_for_user(canonical, session_language)
    _store_display_to_canonical_action_map(db, canonical, display_result)

    db.log_event("CHAT", {"role": "User", "content": req.message})
    db.log_event(
        "CHAT",
        {
            "role": "Keeper",
            "content": canonical.get("narrative", ""),
            "display_content": display_result.get("narrative", ""),
        },
    )
    return display_result


def _apply_dead_pc_note_if_needed(db: SessionDB, req: ChatRequest) -> ChatRequest:
    dead_pcs = [a for a in db.list_actors("PC") if a.get("status") in ("dead", "insane")]
    if not dead_pcs:
        return req

    names = ", ".join(a["name"] for a in dead_pcs)
    note = (
        f"\n\n[KEEPER NOTE: {names} are dead/incapacitated and cannot act. "
        "Do not narrate their actions. Acknowledge their fate if relevant.]"
    )
    return _clone_request_with_message(req, req.message + note)


async def handle_chat_logic(req: ChatRequest) -> dict:
    db = _ensure_session_db(req.session_id)

    intercepted = await _maybe_intercept_roll_gate(db, req)
    if intercepted is not None:
        return _finalize_intercepted_result(db, req, intercepted)

    db.log_event("CHAT", {"role": "User", "content": req.message})
    req = _apply_dead_pc_note_if_needed(db, req)

    canonical_action = _canonical_action_for_keeper(db, req.message)
    turn = _load_turn_context(db, req, canonical_action)

    llm = get_llm(
        temperature=req.temperature,
        task="chat_text",
        num_ctx=req.num_ctx,
        json_mode=False,
    )
    chain = PromptTemplate.from_template(turn["keeper_system_prompt"]) | llm

    response_text = chain.invoke(turn["prompt_vars"])
    logger.info("handle_chat_logic(): raw LLM response preview: %r", str(response_text)[:4000])

    result = _parse_and_postprocess_result(
        db=db,
        req=req,
        canonical_action=canonical_action,
        raw_text=str(response_text),
        chain=chain,
        turn=turn,
    )

    return _finalize_keeper_result(
        db=db,
        req=req,
        result=result,
        session_language=turn["session_language"],
        setting_override=turn["setting_override"],
        era_override=turn["era_override"],
        cur=turn["cur"],
    )


async def stream_chat_logic(req: ChatRequest):
    db = _ensure_session_db(req.session_id)

    intercepted = await _maybe_intercept_roll_gate(db, req)
    if intercepted is not None:
        display_intercepted = _finalize_intercepted_result(db, req, intercepted)
        yield f"data: {json.dumps({'type': 'done', 'payload': display_intercepted}, ensure_ascii=False)}\n\n"
        return

    db.log_event("CHAT", {"role": "User", "content": req.message})
    req = _apply_dead_pc_note_if_needed(db, req)

    canonical_action = _canonical_action_for_keeper(db, req.message)
    turn = _load_turn_context(db, req, canonical_action)

    llm = get_llm(
        temperature=req.temperature,
        task="chat_text",
        num_ctx=req.num_ctx,
        json_mode=False,
    )
    chain = PromptTemplate.from_template(turn["keeper_system_prompt"]) | llm

    # Do not stream raw internal English JSON tokens to the UI.
    # The user-facing payload is localized only after parsing/finalization.
    full_text = ""
    async for chunk in chain.astream(turn["prompt_vars"]):
        full_text += str(chunk)

    logger.info("stream_chat_logic(): full streamed LLM response preview: %r", full_text[:4000])

    result = _parse_and_postprocess_result(
        db=db,
        req=req,
        canonical_action=canonical_action,
        raw_text=full_text,
        chain=chain,
        turn=turn,
    )

    display_result = _finalize_keeper_result(
        db=db,
        req=req,
        result=result,
        session_language=turn["session_language"],
        setting_override=turn["setting_override"],
        era_override=turn["era_override"],
        cur=turn["cur"],
    )

    yield f"data: {json.dumps({'type': 'done', 'payload': display_result}, ensure_ascii=False)}\n\n"
