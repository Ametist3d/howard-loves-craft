import asyncio
import json
import logging

from langchain_core.prompts import PromptTemplate

from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import ChatRequest
from utils.helpers import (
    build_verdict_guard,
    extract_json,
    extract_last_turn_ban,
    get_chat_history,
    get_language_name,
    get_llm,
    has_roll_verdict,
    normalize_language_code,
)
from utils.helper_actions import clear_pending_roll, intercept_player_action_for_roll_gate, save_pending_roll
from utils.helper_story import build_scene_loop_guard, build_scene_prompt, build_stall_forcing_guard, compress_story
from utils.helper_state import (
    apply_state_updates,
    assemble_keeper_prompt,
    build_authoritative_context,
    looks_like_valid_keeper_response,
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
    scen_db,
)
from utils.engine_session import generate_opening_scene_logic
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL


async def handle_chat_logic(req: ChatRequest) -> dict:
    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    intercepted = None

    if has_roll_verdict(req.message):
        clear_pending_roll(db)
    else:
        req_lower = req.message.lower()
        is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
        is_system_msg = req.message.strip().startswith("[SYSTEM")

        if not is_start_msg and not is_system_msg:
            intercepted = intercept_player_action_for_roll_gate(db, req.message)

    if intercepted is not None:
        db.log_event("CHAT", {"role": "User", "content": req.message})
        db.log_event("CHAT", {"role": "Keeper", "content": intercepted.get("narrative", "")})
        return intercepted

    db.log_event("CHAT", {"role": "User", "content": req.message})

    dead_pcs = [a for a in db.list_actors("PC") if a.get("status") in ("dead", "insane")]
    if dead_pcs:
        names = ", ".join(a["name"] for a in dead_pcs)
        req = req.model_copy(update={
            "message": req.message + (
                f"\n\n[KEEPER NOTE: {names} are dead/incapacitated and cannot act. "
                f"Do not narrate their actions. Acknowledge their fate if relevant.]"
            )
        })

    cur = db.conn.cursor()
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_atoms'")
    row_atoms = cur.fetchone()
    campaign_atoms = row_atoms["value"] if row_atoms else ""

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_themes'")
    row_themes = cur.fetchone()
    themes = row_themes["value"] if row_themes else "STANDARD"

    context_str, state_str = build_authoritative_context(
        db,
        campaign_atoms=campaign_atoms,
        themes=themes,
    )

    req_lower = req.message.lower()
    is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
    combat_turn = _is_combat_turn(db, req.message)

    if req.rag_enabled and rules_db and scen_db and not is_start_msg:
        rule_query = req.message
        if combat_turn:
            rule_query = (
                f"{req.message}\n"
                "Call of Cthulhu 7e combat sequence DEX order melee fight back dodge "
                "firearms dive for cover readied firearm point blank outnumbered "
                "major wound unconscious dying instant death"
            )

        all_docs = (
            rules_db.similarity_search(rule_query, k=req.top_k)
            + scen_db.similarity_search(req.message, k=req.top_k)
        )
        if all_docs:
            context_str += "\n\n--- RELEVANT RULEBOOK/SCENARIO LORE ---\n"
            context_str += "\n\n".join([doc.page_content for doc in all_docs])

    llm = get_llm(temperature=req.temperature, num_ctx=req.num_ctx)

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row_setting = cur.fetchone()
    setting_override = row_setting["value"] if row_setting else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row_era = cur.fetchone()
    era_override = row_era["value"] if row_era else "1920s Lovecraftian Horror"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row_lang = cur.fetchone()
    session_language = normalize_language_code(row_lang["value"] if row_lang else "en")

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row_prompt_dir = cur.fetchone()
    prompt_dir = row_prompt_dir["value"] if row_prompt_dir else None

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=has_roll_verdict(req.message),
        include_scene_progression=True,
        include_opening_scene=is_start_msg,
        prompt_dir=prompt_dir,
    )

    chain = PromptTemplate.from_template(keeper_system_prompt) | llm

    scene_loop_guard = build_scene_loop_guard(db)
    verdict_guard = build_verdict_guard(req.message)
    stall_guard = build_stall_forcing_guard(db)

    combat_state = get_combat_state(db) if combat_turn else {}
    combat_state_text = json.dumps(combat_state, ensure_ascii=False)

    response_text = chain.invoke({
        "language": session_language,
        "language_name": get_language_name(session_language),
        "campaign_context": (
            context_str
            + "\n\n--- CURRENT GAME STATE ---\n"
            + state_str
            + "\n\n--- CURRENT COMBAT STATE ---\n"
            + combat_state_text
            + "\n\n"
            + scene_loop_guard
            + ("\n\n" + stall_guard if stall_guard else "")
            + ("\n\n" + verdict_guard if verdict_guard else "")
        ),
        "era_context": setting_override + " " + era_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": extract_last_turn_ban(db),
    })

    logger.info("handle_chat_logic(): raw LLM response preview: %r", str(response_text)[:4000])
    result = extract_json(response_text)

    if not looks_like_valid_keeper_response(str(response_text), result):
        logger.warning("Non-contract LLM reply detected; attempting one repair regeneration")

        repair_prompt = (
            "Your previous reply violated the required Keeper output contract.\n"
            "Return ONLY a valid <SYSTEM_RESPONSE_JSON>...</SYSTEM_RESPONSE_JSON> block.\n"
            "Do not ask the user for clarification.\n"
            "Do not explain your answer.\n"
            "Do not output markdown.\n"
            "Preserve the same current scene, current action, and current game state.\n"
        )

        repaired_text = chain.invoke({
            "language": session_language,
            "language_name": get_language_name(session_language),
            "campaign_context": (
                context_str
                + "\n\n--- CURRENT GAME STATE ---\n"
                + state_str
                + "\n\n--- CURRENT COMBAT STATE ---\n"
                + combat_state_text
                + "\n\n"
                + scene_loop_guard
                + ("\n\n" + stall_guard if stall_guard else "")
                + ("\n\n" + verdict_guard if verdict_guard else "")
                + "\n\n--- CONTRACT REPAIR NOTICE ---\n"
                + repair_prompt
            ),
            "era_context": setting_override + " " + era_override,
            "history": get_chat_history(db, limit=15),
            "action": req.message,
            "last_turn_ban": extract_last_turn_ban(db),
        })

        logger.warning("Repair regeneration raw preview: %r", str(repaired_text)[:2000])
        repaired_result = extract_json(repaired_text)

        if looks_like_valid_keeper_response(str(repaired_text), repaired_result):
            result = repaired_result

    if combat_turn:
        result = resolve_combat_turn(db, result)

    violations = validate_llm_response_against_state(db, result)
    if violations:
        logger.warning("LLM response validation failed: %s", violations)
        db.log_event("LLM_VALIDATION_FAIL", {
            "violations": violations,
            "narrative": result.get("narrative", "")[:500]
        })
        result = sanitize_llm_result_on_validation_failure(result, violations)

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    import asyncio, uuid as _uuid
    from pathlib import Path as _Path

    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    gen_id = _uuid.uuid4().hex
    _image_results[gen_id] = "pending"
    result["generation_id"] = gen_id
    result["image_url"] = None

    async def _generate_image_bg(gid: str, narrative: str, setting: str, era: str):
        try:
            from img_gen.comfy_client import ComfyClient
            _comfy = ComfyClient(_COMFY_BASE_URL)

            visual_history = ""
            char_visuals = ""
            if req.session_id in active_dbs:
                _db = active_dbs[req.session_id]
                _cur = _db.conn.cursor()
                _cur.execute("SELECT value FROM kv_store WHERE key='visual_history'")
                row = _cur.fetchone()
                visual_history = row["value"] if row else ""

                _cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'")
                rows = _cur.fetchall()
                if rows:
                    char_visuals = "ESTABLISHED CHARACTERS (maintain consistent appearance):\n"
                    char_visuals += "\n".join(f"- {r['value']}" for r in rows)

            scene_prompt = build_scene_prompt(
                narrative, era=era, setting=setting,
                visual_history=visual_history,
                char_visuals=char_visuals
            )
            logger.info(f"Image prompt [{gid}]: {scene_prompt}")

            with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as _f:
                _body = json.load(_f)
            _body["params"]["prompt"] = scene_prompt

            img_result = _comfy.generate(_body)
            _image_results[gid] = img_result["image_url"]
            logger.info(f"Image ready [{gid}]: {img_result['image_url']}")
        except Exception as _e:
            logger.warning(f"Image generation failed [{gid}]: {_e}")
            _image_results[gid] = None

    asyncio.create_task(_generate_image_bg(
        gen_id,
        result.get("narrative", ""),
        setting=setting_override,
        era=era_override
    ))

    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})

    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row_tc = cur.fetchone()
    turn_count = int(row_tc["value"]) + 1 if row_tc else 1
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)", (str(turn_count),))
    db.conn.commit()

    if _should_refresh_digest(turn_count):
        asyncio.create_task(compress_story(db))
        logger.info(f"Story compression triggered at turn {turn_count}")

    apply_state_updates(db, result)
    return result


async def stream_chat_logic(req: ChatRequest):
    import asyncio, json as _json

    if req.session_id not in active_dbs:
        info = create_session_db_file(SESSIONS_DIR, "Fallback Session", "Standard")
        active_dbs[req.session_id] = SessionDB(info.db_path)

    db = active_dbs[req.session_id]
    intercepted = None

    if has_roll_verdict(req.message):
        clear_pending_roll(db)
    else:
        req_lower = req.message.lower()
        is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
        is_system_msg = req.message.strip().startswith("[SYSTEM")

        if not is_start_msg and not is_system_msg:
            intercepted = intercept_player_action_for_roll_gate(db, req.message)

    if intercepted is not None:
        db.log_event("CHAT", {"role": "User", "content": req.message})
        db.log_event("CHAT", {"role": "Keeper", "content": intercepted.get("narrative", "")})
        yield f"data: {_json.dumps({'type': 'done', 'payload': intercepted}, ensure_ascii=False)}\n\n"
        return

    db.log_event("CHAT", {"role": "User", "content": req.message})

    cur = db.conn.cursor()

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_atoms'")
    row = cur.fetchone()
    campaign_atoms = row["value"] if row else ""

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_themes'")
    row = cur.fetchone()
    themes = row["value"] if row else "STANDARD"

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row = cur.fetchone()
    setting_override = row["value"] if row else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row = cur.fetchone()
    era_override = row["value"] if row else "1920s Lovecraftian Horror"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row = cur.fetchone()
    session_language = normalize_language_code(row["value"] if row else "en")

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row = cur.fetchone()
    prompt_dir = row["value"] if row else None

    context_str, state_str = build_authoritative_context(
        db,
        campaign_atoms=campaign_atoms,
        themes=themes,
    )

    req_lower = req.message.lower()
    is_start_msg = "start the story" in req_lower or "open with the first playable scene" in req_lower
    combat_turn = _is_combat_turn(db, req.message)

    if req.rag_enabled and rules_db and scen_db and not is_start_msg:
        rule_query = req.message
        if combat_turn:
            rule_query = (
                f"{req.message}\n"
                "Call of Cthulhu 7e combat sequence DEX order melee fight back dodge "
                "firearms dive for cover readied firearm point blank outnumbered "
                "major wound unconscious dying instant death"
            )

        all_docs = (
            rules_db.similarity_search(rule_query, k=req.top_k)
            + scen_db.similarity_search(req.message, k=req.top_k)
        )
        if all_docs:
            context_str += "\n\n--- RELEVANT RULEBOOK/SCENARIO LORE ---\n"
            context_str += "\n\n".join([doc.page_content for doc in all_docs])

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=has_roll_verdict(req.message),
        include_scene_progression=True,
        include_opening_scene=is_start_msg,
        prompt_dir=prompt_dir,
    )

    llm = get_llm(temperature=req.temperature, num_ctx=req.num_ctx)
    chain = PromptTemplate.from_template(keeper_system_prompt) | llm

    verdict_guard = build_verdict_guard(req.message)
    scene_loop_guard = build_scene_loop_guard(db)
    stall_guard = build_stall_forcing_guard(db)

    combat_state = get_combat_state(db) if combat_turn else {}
    combat_state_text = json.dumps(combat_state, ensure_ascii=False)

    prompt_vars = {
        "language": session_language,
        "language_name": get_language_name(session_language),
        "campaign_context": (
            context_str
            + "\n\n--- CURRENT GAME STATE ---\n"
            + state_str
            + "\n\n--- CURRENT COMBAT STATE ---\n"
            + combat_state_text
            + "\n\n"
            + scene_loop_guard
            + ("\n\n" + stall_guard if stall_guard else "")
            + ("\n\n" + verdict_guard if verdict_guard else "")
        ),
        "era_context": setting_override + " " + era_override,
        "history": get_chat_history(db, limit=15),
        "action": req.message,
        "last_turn_ban": extract_last_turn_ban(db),
    }

    full_text = ""
    async for chunk in chain.astream(prompt_vars):
        full_text += chunk
        yield f"data: {_json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"

    logger.info("stream_chat_logic(): full streamed LLM response preview: %r", full_text[:4000])

    result = extract_json(full_text)

    if not looks_like_valid_keeper_response(str(full_text), result):
        logger.warning("Non-contract LLM reply detected; attempting one repair regeneration")

        repair_prompt = (
            "Your previous reply violated the required Keeper output contract.\n"
            "Return ONLY a valid <SYSTEM_RESPONSE_JSON>...</SYSTEM_RESPONSE_JSON> block.\n"
            "Do not ask the user for clarification.\n"
            "Do not explain your answer.\n"
            "Do not output markdown.\n"
            "Preserve the same current scene, current action, and current game state.\n"
        )

        repaired_text = chain.invoke({
            "language": session_language,
            "language_name": get_language_name(session_language),
            "campaign_context": (
                context_str
                + "\n\n--- CURRENT GAME STATE ---\n"
                + state_str
                + "\n\n--- CURRENT COMBAT STATE ---\n"
                + combat_state_text
                + "\n\n"
                + scene_loop_guard
                + ("\n\n" + stall_guard if stall_guard else "")
                + ("\n\n" + verdict_guard if verdict_guard else "")
                + "\n\n--- CONTRACT REPAIR NOTICE ---\n"
                + repair_prompt
            ),
            "era_context": setting_override + " " + era_override,
            "history": get_chat_history(db, limit=15),
            "action": req.message,
            "last_turn_ban": extract_last_turn_ban(db),
        })

        logger.warning("Repair regeneration raw preview: %r", str(repaired_text)[:2000])
        repaired_result = extract_json(repaired_text)

        if looks_like_valid_keeper_response(str(repaired_text), repaired_result):
            result = repaired_result

    if combat_turn:
        result = resolve_combat_turn(db, result)

    violations = validate_llm_response_against_state(db, result)
    if violations:
        logger.warning("LLM response validation failed: %s", violations)
        db.log_event("LLM_VALIDATION_FAIL", {
            "violations": violations,
            "narrative": result.get("narrative", "")[:500]
        })
        result = sanitize_llm_result_on_validation_failure(result, violations)

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    from pathlib import Path as _Path
    import uuid as _uuid

    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    gen_id = _uuid.uuid4().hex
    _image_results[gen_id] = "pending"
    result["generation_id"] = gen_id
    result["image_url"] = None

    async def _generate_image_bg(gid, narrative, setting, era):
        try:
            from img_gen.comfy_client import ComfyClient
            _comfy = ComfyClient(_COMFY_BASE_URL)

            visual_history, char_visuals = "", ""
            if req.session_id in active_dbs:
                _db = active_dbs[req.session_id]
                _cur = _db.conn.cursor()
                _cur.execute("SELECT value FROM kv_store WHERE key='visual_history'")
                r = _cur.fetchone()
                visual_history = r["value"] if r else ""

                _cur.execute("SELECT key, value FROM kv_store WHERE key LIKE 'char_visual_%'")
                rows = _cur.fetchall()
                if rows:
                    char_visuals = "ESTABLISHED CHARACTERS:\n" + "\n".join(f"- {r['value']}" for r in rows)

            scene_prompt = build_scene_prompt(
                narrative,
                era=era,
                setting=setting,
                visual_history=visual_history,
                char_visuals=char_visuals
            )

            with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as f:
                body = _json.load(f)
            body["params"]["prompt"] = scene_prompt

            img_result = _comfy.generate(body)
            _image_results[gid] = img_result["image_url"]
        except Exception as e:
            logger.warning(f"Image generation failed [{gid}]: {e}")
            _image_results[gid] = None

    asyncio.create_task(_generate_image_bg(
        gen_id,
        result.get("narrative", ""),
        setting_override,
        era_override
    ))

    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})

    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row = cur.fetchone()
    turn_count = int(row["value"]) + 1 if row else 1
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)", (str(turn_count),))
    db.conn.commit()

    if _should_refresh_digest(turn_count):
        asyncio.create_task(compress_story(db))

    apply_state_updates(db, result)
    yield f"data: {_json.dumps({'type': 'done', 'payload': result}, ensure_ascii=False)}\n\n"
