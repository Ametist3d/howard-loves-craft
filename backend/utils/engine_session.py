import json
import os
import random
import re
import asyncio

from langchain_core.prompts import PromptTemplate

#pylint: disable=import-error
from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest
from utils.helpers import (
    extract_blueprint_json,
    extract_json,
    get_llm,
    _normalize_repaired_blueprint,
    read_prompt,
)
from utils.helper_story import (
    repaired_prompt,
    local_composed_template,
    _seed_terms,
    _setting_terms,
    _doc_setting_coherence,
    _select_coherent_docs,
    _extract_strict_json_object,
    _build_deterministic_opening_result,
    _build_blueprint_scaffold,
    _generate_blueprint_header,
    _generate_act_payload,
    _normalize_act_payload,
)
from utils.helper_actions import clear_pending_roll, save_pending_roll
from utils.helper_state import (
    apply_state_updates,
    assemble_keeper_prompt,
    build_authoritative_context,
    build_opening_fallback_result,
    validate_opening_scene_response,
)
from utils.prompt_translate import (
    get_language_name,
    normalize_language_code,
    ensure_translated_prompts,
    translate_opening_result_for_user,
    translate_scenario_summary_for_user,
    translate_blueprint_for_display,
)
from utils.engine import (
    SESSIONS_DIR,
    _dbg,
    _derive_initial_objective,
    _trace_session,
    _trace_session_json,
    active_dbs,
    logger,
    scen_db,
    _image_results,
)
from utils.engine_chat import _generate_image_bg
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

# ----------------------------
# Prompt prep / LLM helpers
# ----------------------------

def _prepare_multicall_prompt_dir(session_id: str, language: str) -> str:
    # Keep runtime prompts in English for local-model stability.
    # User-facing localization should happen later in the chat flow, not in start-session.
    return ensure_translated_prompts(
        session_id,
        "en",
        filenames=(
            "character_gen.txt",
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
        ),
    )


def _opening_llm(temperature: float):
    return get_llm(
        temperature=temperature,
        task="opening_json",
    )


def _synth_language_vars(user_lang: str) -> tuple[str, str]:
    # Internal scenario generation stays in English.
    return "en", "English"


def _render_literal_prompt(template: str, mapping: dict[str, object]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"__{key.upper()}__", str(value))
    return out

def _doc_text_blob(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    parts = [
        doc.page_content or "",
        str(meta.get("abstraction", "") or ""),
        str(meta.get("title_en", "") or ""),
        str(meta.get("display_name", "") or ""),
        str(meta.get("Header_2", "") or ""),
        str(meta.get("aliases", "") or ""),
        str(meta.get("archetype", "") or ""),
        str(meta.get("role", "") or ""),
        str(meta.get("type", "") or ""),
    ]
    return " ".join(parts).lower()

# ----------------------------
# Generic JSON / structure helpers
# ----------------------------


def _choose_structure_budget(rng: random.Random | None = None) -> tuple[int, int, list[int]]:
    rng = rng or random.Random()

    total_scenes = rng.randint(6, 10)
    min_act_count = max(2, (total_scenes + 3) // 4)  # no act > 4 scenes
    max_act_count = min(4, total_scenes)
    act_count = rng.randint(min_act_count, max_act_count)

    base = total_scenes // act_count
    remainder = total_scenes % act_count
    scene_counts = [base + (1 if i < remainder else 0) for i in range(act_count)]
    return act_count, total_scenes, scene_counts


def _validate_plan_json(plan: dict, *, expected_act_count: int | None = None) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("Planner output is not a JSON object")

    hook_type = str(plan.get("hook_type", "") or "").strip()
    if not hook_type:
        plan["hook_type"] = "anomaly_report"

    modules = plan.get("act_module_types")
    if not isinstance(modules, list):
        raise ValueError("Planner output must contain act_module_types list")

    modules = [str(x).strip() for x in modules if str(x).strip()]
    if expected_act_count is not None and len(modules) != expected_act_count:
        raise ValueError(
            f"Planner output must contain exactly {expected_act_count} act_module_types, got {len(modules)}"
        )
    if expected_act_count is None and not (2 <= len(modules) <= 4):
        raise ValueError("Planner output must contain 2-4 act_module_types")
    plan["act_module_types"] = modules

    allowed_resolution_types = {
        "suppress",
        "expose",
        "contain",
        "bargain",
        "evacuate",
        "misdirect",
        "sever",
        "sacrifice",
        "transfer_cost",
        "delay",
        "destroy_with_cost",
        "human_authority_solution",
        "occult_or_technical_override",
        "escape_and_bury",
    }
    resolution_types = plan.get("resolution_types")
    if not isinstance(resolution_types, list):
        resolution_types = []
    resolution_types = [
        str(x).strip()
        for x in resolution_types
        if str(x).strip() in allowed_resolution_types
    ]
    for fallback in ("contain", "expose", "delay"):
        if len(resolution_types) >= 2:
            break
        if fallback not in resolution_types:
            resolution_types.append(fallback)
    plan["resolution_types"] = resolution_types[:3]

    defaults_map = {
        "false_leads": [
            "A mundane technical explanation seems plausible at first.",
            "An institutional or human rival appears responsible.",
        ],
        "contradictions": [
            "One clue directly conflicts with the surface explanation.",
            "Timing or physical evidence does not match the official story.",
        ],
        "reversals": [
            "The apparent cause is actually a symptom.",
            "The presumed victim or target is part of the deeper mechanism.",
        ],
        "dynamic_pressures": [
            "Authority pressure pushes investigators toward the wrong conclusion.",
            "Time, environment, or system instability keeps getting worse.",
        ],
    }
    for key, defaults in defaults_map.items():
        value = plan.get(key)
        if not isinstance(value, list):
            value = []
        value = [str(x).strip() for x in value if str(x).strip()]
        for fallback in defaults:
            if len(value) >= 2:
                break
            value.append(fallback)
        plan[key] = value[:3]

    return plan


def _validate_blueprint_structure(
    blueprint: dict,
    *,
    expected_act_count: int,
    expected_scene_counts: list[int],
) -> dict:
    if not isinstance(blueprint, dict):
        raise ValueError("Blueprint output is not a JSON object")

    acts = blueprint.get("acts")
    if not isinstance(acts, list) or len(acts) != expected_act_count:
        raise ValueError(f"Blueprint must contain exactly {expected_act_count} acts")

    for idx, (act, expected_scene_count) in enumerate(zip(acts, expected_scene_counts), start=1):
        scenes = act.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != expected_scene_count:
            raise ValueError(
                f"Blueprint act {idx} must contain exactly {expected_scene_count} scenes"
            )

    return blueprint

def _fallback_hook_type(seed: str) -> str:
    s = (seed or "").lower()
    if any(x in s for x in ["missing", "disappear", "vanish"]):
        return "disappearance"
    if any(x in s for x in ["radio", "signal", "broadcast", "transmission"]):
        return "strange_signal"
    return "unusual_event"

def _fallback_plan(act_count: int, seed: str) -> dict:
    base_modules = [
        "surface_inquiry",
        "danger_probe",
        "contradiction_reveal",
        "costly_climax",
    ]
    modules = base_modules[:act_count]
    if len(modules) < act_count:
        modules += ["danger_probe"] * (act_count - len(modules))

    return {
        "hook_type": _fallback_hook_type(seed),
        "act_module_types": modules,
        "resolution_types": ["contain", "expose"],
        "false_leads": [
            "A mundane technical explanation seems plausible at first.",
            "A human rival or institutional cover story appears responsible."
        ],
        "contradictions": [
            "Evidence conflicts with the surface explanation.",
            "Timing or physical traces do not fit the official account."
        ],
        "reversals": [
            "The apparent cause is actually a symptom.",
            "The presumed victim or target is part of the deeper mechanism."
        ],
        "dynamic_pressures": [
            "Authority pressure pushes investigators toward the wrong conclusion.",
            "Time or environmental instability keeps getting worse."
        ],
    }

# ----------------------------
# Scenario synthesis helpers
# ----------------------------

def _build_scenario_summary_from_blueprint(blueprint: dict) -> str:
    return (
        f"SCENARIO TITLE: {blueprint.get('title', 'Unknown')}\n"
        f"SETTING: {blueprint.get('era_and_setting', '')}\n"
        f"HOOK: {blueprint.get('inciting_hook', '')}\n"
        f"CORE MYSTERY: {blueprint.get('core_mystery', '')}\n"
        f"ATMOSPHERE: {blueprint.get('atmosphere_notes', '')}"
    )

def _dict_items_only(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]

def _serialize_doc_for_trace(doc) -> dict:
    meta = getattr(doc, "metadata", {}) or {}
    return {
        "metadata": meta,
        "page_content": (doc.page_content or "")[:4000],
    }

def _condense_atom(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}

    title = str(meta.get("title_en") or meta.get("display_name") or meta.get("Header_2") or "Unknown").strip()
    atom_type = str(meta.get("type") or "").strip().lower()
    role = str(meta.get("role") or "").strip().lower()
    abstraction = str(meta.get("abstraction") or "").strip()

    # keep atom representation minimal and stable
    lines = [
        f"title: {title[:80]}",
        f"type: {atom_type[:40]}",
        f"role: {role[:60]}",
    ]

    if abstraction:
        abstraction = re.sub(r"\s+", " ", abstraction).strip()
        lines.append(f"abstraction: {abstraction[:180]}")

    return "\n".join(lines)
    
def _run_multi_call_scenario_synth(
    *,
    db: SessionDB,
    prompt_dir: str,
    themes_str: str,
    era_context: str,
    lang: str,
    query_text: str,
    raw_atoms_text: str,
) -> tuple[dict, str]:
    synth_lang, synth_language_name = _synth_language_vars(lang)

    act_count, total_scenes, scene_counts = _choose_structure_budget()
    scene_count_plan = json.dumps(scene_counts, ensure_ascii=False)

    if len(raw_atoms_text) > 1800:
        raw_atoms_text = raw_atoms_text[:1800]

    base_vars = {
        "themes": themes_str,
        "era_context": era_context,
        "language": synth_lang,
        "language_name": synth_language_name,
        "seed": query_text,
        "atoms": raw_atoms_text,
        "target_act_count": act_count,
        "target_total_scenes": total_scenes,
        "scene_count_plan": scene_count_plan,
    }

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    logger.info(
        "SCENARIO_SYNTH: planner invoke starting | acts=%s | total_scenes=%s | scene_counts=%s",
        act_count,
        total_scenes,
        scene_counts,
    )
    _trace_session_json(db, "SCENARIO_STRUCTURE_BUDGET", {
        "provider": provider,
        "act_count": act_count,
        "total_scenes": total_scenes,
        "scene_counts": scene_counts,
        "themes": themes_str,
        "era_context": era_context,
        "query_text": query_text,
    })

    # ------------------------------------------------------------------
    # OLLAMA PATH: deterministic scaffold + tiny bounded JSON generations
    # ------------------------------------------------------------------
    if provider == "ollama":
        logger.info("SCENARIO_SYNTH: using Ollama scaffold/tagged path")

        plan = _validate_plan_json(
            _fallback_plan(act_count, query_text),
            expected_act_count=act_count,
        )
        _trace_session_json(db, "SCENARIO_PLAN_FALLBACK_USED", plan)

        blueprint = _build_blueprint_scaffold(
            act_count=act_count,
            scene_counts=scene_counts,
            era_context=era_context,
            seed=query_text,
            plan=plan,
        )

        try:
            header = _generate_blueprint_header(
                era_context=era_context,
                seed=query_text,
                atoms=raw_atoms_text,
                plan=plan,
            )
        except Exception as e:
            logger.warning("SCENARIO_SYNTH: header generation failed, using defaults: %s", e)
            header = {}

        blueprint["title"] = str(header.get("TITLE") or "Untitled Scenario")
        blueprint["era_and_setting"] = str(header.get("ERA_AND_SETTING") or era_context)
        blueprint["atmosphere_notes"] = str(header.get("ATMOSPHERE_NOTES") or "Tense, uncanny, and escalating.")
        blueprint["inciting_hook"] = str(header.get("INCITING_HOOK") or query_text)
        blueprint["core_mystery"] = str(header.get("CORE_MYSTERY") or "Something hidden is reshaping perception and behavior.")
        blueprint["hidden_threat"] = str(header.get("HIDDEN_THREAT") or "A concealed predatory force is exploiting the environment.")
        blueprint["truth_the_players_never_suspect"] = str(
            header.get("TRUTH_THE_PLAYERS_NEVER_SUSPECT")
            or "The anomaly is active, strategic, and not merely environmental."
        )

        blueprint["scenario_engine"]["surface_explanation"] = str(
            header.get("SURFACE_EXPLANATION") or blueprint["scenario_engine"]["surface_explanation"]
        )
        blueprint["scenario_engine"]["actual_explanation"] = str(
            header.get("ACTUAL_EXPLANATION") or blueprint["scenario_engine"]["actual_explanation"]
        )

        blueprint["scenario_engine"]["climax_choices"] = [
            {
                "option": str(header.get("CLIMAX_1_OPTION") or "Contain the threat"),
                "cost": str(header.get("CLIMAX_1_COST") or "Immediate personal loss"),
                "consequence": str(header.get("CLIMAX_1_CONSEQUENCE") or "The danger is reduced, but the cost remains."),
            },
            {
                "option": str(header.get("CLIMAX_2_OPTION") or "Expose the truth"),
                "cost": str(header.get("CLIMAX_2_COST") or "Escalation and retaliation"),
                "consequence": str(header.get("CLIMAX_2_CONSEQUENCE") or "The hidden pattern becomes visible, but fallout spreads."),
            },
        ]

        previous_acts_summary = "No previous acts yet."
        for i, expected_scene_count in enumerate(scene_counts):
            act_no = i + 1
            module_type = plan["act_module_types"][i]

            try:
                act_payload = _generate_act_payload(
                    act_no=act_no,
                    module_type=module_type,
                    scene_count=expected_scene_count,
                    seed=query_text,
                    era_context=era_context,
                    atoms=raw_atoms_text,
                    plan=plan,
                    previous_acts_summary=previous_acts_summary,
                )
            except Exception as e:
                logger.warning("SCENARIO_SYNTH: act %s generation failed, using fallback: %s", act_no, e)
                act_payload = {}

            act_payload = _normalize_act_payload(
                act_payload,
                act_no=act_no,
                scene_count=expected_scene_count,
                module_type=module_type,
            )

            blueprint["acts"][i] = act_payload
            previous_acts_summary += f"\nAct {act_no}: {act_payload['summary']}"

        blueprint = _validate_blueprint_structure(
            blueprint,
            expected_act_count=act_count,
            expected_scene_counts=scene_counts,
        )
        _trace_session_json(db, "SCENARIO_BLUEPRINT_FINAL_OLLAMA", blueprint)
        return blueprint, _build_scenario_summary_from_blueprint(blueprint)

    # ------------------------------------------------------------------
    # NON-OLLAMA PATH: keep the full strong-model JSON compose flow
    # ------------------------------------------------------------------
    plan_template = read_prompt("scenario/scenario_plan.txt", prompt_dir=prompt_dir)
    plan_prompt = PromptTemplate.from_template(plan_template).format(**base_vars)
    _trace_session(db, "SCENARIO_PLAN_PROMPT", plan_prompt)

    plan_chain = PromptTemplate.from_template(plan_template) | get_llm(
        temperature=0.08,
        num_ctx=8192,
        num_predict=1200,
        json_mode=True,
    )
    plan_raw = plan_chain.invoke(base_vars)
    _trace_session(db, "SCENARIO_PLAN_RAW", str(plan_raw))
    logger.info("SCENARIO_SYNTH: planner invoke finished")

    try:
        plan = _validate_plan_json(
            _extract_strict_json_object(plan_raw),
            expected_act_count=act_count,
        )
    except Exception as e:
        logger.warning("SCENARIO_SYNTH: planner parse failed (%s), using deterministic fallback plan", e)
        _trace_session(db, "SCENARIO_PLAN_PARSE_ERROR", repr(e))
        plan = _validate_plan_json(
            _fallback_plan(act_count, query_text),
            expected_act_count=act_count,
        )
        _trace_session_json(db, "SCENARIO_PLAN_FALLBACK_USED", plan)

    logger.info("SCENARIO_SYNTH: compose invoke starting")
    compose_template = read_prompt("scenario/scenario_compose.txt", prompt_dir=prompt_dir)

    compose_vars = dict(base_vars)
    compose_vars["plan_json"] = json.dumps(plan, ensure_ascii=False)

    max_compose_attempts = 2
    blueprint = None
    compose_raw = ""

    for attempt in range(1, max_compose_attempts + 1):
        local_compose_vars = dict(compose_vars)

        if attempt == 1:
            local_compose_template = compose_template
            compose_temp = 0.15
            compose_num_predict = int(os.getenv("SCENARIO_SYNTH_NUM_PREDICT", "4096"))

            compose_prompt_rendered = PromptTemplate.from_template(local_compose_template).format(**local_compose_vars)
            compose_raw = (
                PromptTemplate.from_template(local_compose_template)
                | get_llm(
                    temperature=compose_temp,
                    num_ctx=int(os.getenv("SCENARIO_SYNTH_NUM_CTX", "16384")),
                    num_predict=compose_num_predict,
                    json_mode=True,
                )
            ).invoke(local_compose_vars)
        else:
            local_compose_template = compose_template
            compose_temp = 0.08
            compose_num_predict = 2600

            compose_prompt_rendered = PromptTemplate.from_template(local_compose_template).format(**local_compose_vars)
            compose_raw = (
                PromptTemplate.from_template(local_compose_template)
                | get_llm(
                    temperature=compose_temp,
                    num_ctx=int(os.getenv("SCENARIO_SYNTH_NUM_CTX", "16384")),
                    num_predict=compose_num_predict,
                    json_mode=True,
                )
            ).invoke(local_compose_vars)

        _trace_session(db, f"SCENARIO_COMPOSE_PROMPT_ATTEMPT_{attempt}", compose_prompt_rendered)
        _trace_session(db, f"SCENARIO_COMPOSE_RAW_ATTEMPT_{attempt}", str(compose_raw))
        logger.info(
            "SCENARIO_SYNTH: compose invoke finished (attempt %d/%d)",
            attempt,
            max_compose_attempts,
        )
        _dbg("SCENARIO_COMPOSE RAW", str(compose_raw)[:12000])

        try:
            blueprint = extract_blueprint_json(compose_raw)
            blueprint = _validate_blueprint_structure(
                blueprint,
                expected_act_count=act_count,
                expected_scene_counts=scene_counts,
            )
            break
        except Exception as e:
            _trace_session(db, f"SCENARIO_COMPOSE_ERROR_ATTEMPT_{attempt}", repr(e))
            logger.warning("SCENARIO_SYNTH: compose attempt %d failed: %s", attempt, e)

    if blueprint is None:
        repair_prompt = _render_literal_prompt(repaired_prompt, {
            "act_count": act_count,
            "scene_counts": json.dumps(scene_counts, ensure_ascii=False),
            "compose_raw": compose_raw,
        })

        _trace_session(db, "SCENARIO_COMPOSE_REPAIR_PROMPT", repair_prompt)
        repaired_raw = get_llm(
            temperature=0.05,
            num_ctx=12000,
            num_predict=2400,
            json_mode=True,
        ).invoke(repair_prompt)
        _trace_session(db, "SCENARIO_COMPOSE_REPAIR_RAW", str(repaired_raw))

        try:
            blueprint = extract_blueprint_json(repaired_raw)
            blueprint = _normalize_repaired_blueprint(
                blueprint,
                act_count=act_count,
                scene_counts=scene_counts,
                era_context=era_context,
                seed=query_text,
                plan=plan,
            )
            blueprint = _validate_blueprint_structure(
                blueprint,
                expected_act_count=act_count,
                expected_scene_counts=scene_counts,
            )
        except Exception as e:
            _trace_session(db, "SCENARIO_COMPOSE_FINAL_FAILURE", repr(e))
            raise e

    _trace_session_json(db, "SCENARIO_BLUEPRINT_FINAL_NON_OLLAMA", blueprint)
    return blueprint, _build_scenario_summary_from_blueprint(blueprint)


# ----------------------------
# Character generation
# ----------------------------

async def generate_character_logic(req: CharGenRequest) -> dict:
    session_id = "local_session"
    lang = normalize_language_code(req.language)
    prompt_dir = _prepare_multicall_prompt_dir(session_id, lang)

    llm = get_llm(
        temperature=0.7,
        task="character_json",
    )
    chain = PromptTemplate.from_template(
        read_prompt("character_gen.txt", prompt_dir=prompt_dir)
    ) | llm

    response_text = chain.invoke({
        "language": lang,
        "language_name": get_language_name(lang),
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected",
    })
    return extract_json(response_text)


# ----------------------------
# Opening scene
# ----------------------------

async def generate_opening_scene_logic(db: SessionDB, session_id: str = "local_session") -> dict:
    
    cur = db.conn.cursor()

    campaign_atoms = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='scenario_atoms'"
    ).fetchone()
    campaign_atoms = campaign_atoms[0] if campaign_atoms else ""

    themes_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='scenario_themes'"
    ).fetchone()
    themes = themes_row[0] if themes_row else "STANDARD"

    setting_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='scenario_setting'"
    ).fetchone()
    setting_override = setting_row[0] if setting_row else "Lovecraftian Horror Lore"

    era_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='era_context'"
    ).fetchone()
    era_override = era_row[0] if era_row else "1920s Lovecraftian Horror"

    lang_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='language'"
    ).fetchone()
    session_language = normalize_language_code(lang_row[0] if lang_row else "en")

    current_scene_row = db.conn.execute(
    "SELECT value FROM kv_store WHERE key='current_scene'"
    ).fetchone()
    current_scene_name = current_scene_row[0] if current_scene_row else ""

    current_scene_loc_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='current_scene_location'"
    ).fetchone()
    current_scene_location = current_scene_loc_row[0] if current_scene_loc_row else ""

    prompt_dir_row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='prompt_dir'"
    ).fetchone()
    prompt_dir = prompt_dir_row[0] if prompt_dir_row else None

    context_str, state_str = build_authoritative_context(
        db,
        campaign_atoms=campaign_atoms,
        themes=themes,
        include_next_hint=False,
    )

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=False,
        include_scene_progression=False,
        include_opening_scene=True,
        prompt_dir=prompt_dir,
    )

    opening_lang, opening_language_name = _synth_language_vars(session_language)
    llm = _opening_llm(0.15)
    chain = PromptTemplate.from_template(keeper_system_prompt) | llm

    response_text = chain.invoke({
        "language": opening_lang,
        "language_name": opening_language_name,
        "campaign_context": context_str + "\n\n--- CURRENT GAME STATE ---\n" + state_str,
        "era_context": setting_override + " " + era_override,
        "history": "",
        "action": (
            "Open the scenario with the first playable scene only.\n"
            f"CURRENT SCENE NAME (use this scene): {current_scene_name}\n"
            f"CURRENT SCENE LOCATION (reuse this exact location label in state_updates.location_name): {current_scene_location}\n"
            "Do not skip ahead.\n"
            "Do not move to another scene or another location.\n"
            "state_updates.location_name must match the current scene location exactly.\n"
            "Do not require a roll unless immediate uncertainty is already established.\n"
            "Provide exactly three actionable suggested actions."
        ),
        "last_turn_ban": "",
    })

    logger.info("generate_opening_scene_logic(): raw LLM response preview: %r", str(response_text)[:4000])
    result = extract_json(response_text)

    opening_violations = validate_opening_scene_response(db, result)
    if opening_violations:
        logger.warning("Opening scene validation failed: %s", opening_violations)
        db.log_event("OPENING_VALIDATION_FAIL", {
            "violations": opening_violations,
            "narrative": result.get("narrative", "")[:500],
        })
        result = build_opening_fallback_result(db)

    display_result = translate_opening_result_for_user(result, session_language)
    
    logger.info(
        "Opening translation | lang=%s | narrative_before=%r | narrative_after=%r | actions_before=%r | actions_after=%r",
        session_language,
        result.get("narrative", "")[:180],
        display_result.get("narrative", "")[:180],
        (result.get("suggested_actions") or [])[:3],
        (display_result.get("suggested_actions") or [])[:3],
    )

    import uuid as _uuid
    from pathlib import Path as _Path


    gen_id = _uuid.uuid4().hex
    _image_results[gen_id] = "pending"
    display_result["generation_id"] = gen_id
    display_result["image_url"] = None

    asyncio.create_task(_generate_image_bg(
        generation_id=gen_id,
        session_id=session_id,
        narrative=result.get("narrative", ""),  
        setting=setting_override,
        era=era_override,
    ))

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    db.log_event("SYS_OPENING_SCENE", {"content": "Opening scene generated"})
    db.log_event("CHAT", {"role": "Keeper", "content": display_result.get("narrative", "")})

    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row_tc = cur.fetchone()
    turn_count = int(row_tc[0]) + 1 if row_tc else 1
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)",
        (str(turn_count),),
    )
    db.conn.commit()

    apply_state_updates(db, result)
    return display_result

# ----------------------------
# Session start
# ----------------------------

async def start_session_logic(req: StartSessionRequest) -> dict:
    session_id = "local_session"
    lang = normalize_language_code(req.language)
    prompt_dir = _prepare_multicall_prompt_dir(session_id, lang)

    themes_str = ", ".join(req.themes).upper() if req.themes else "STANDARD"
    era_context = req.era_context or "Cosmic Horror — derive era and aesthetics from the scenario atoms."

    if req.scenarioType == "custom" and req.customPrompt:
        setting_desc = f"Custom: {req.customPrompt[:50]}..."
        query_text = req.customPrompt
    elif req.scenarioType == "prebuilt" and req.picked_seed:
        setting_desc = f"Prebuilt: {req.picked_seed[:50]}..."
        query_text = req.picked_seed
        era_context = req.era_context or era_context
    elif req.picked_seed:
        setting_desc = f"Themes: {themes_str}"
        query_text = req.picked_seed
    else:
        setting_desc = req.scenarioType
        query_text = "Lovecraftian cosmic horror mystery hook and secrets"

    info = create_session_db_file(SESSIONS_DIR, "Call of Cthulhu", setting_desc)
    db = SessionDB(info.db_path)
    active_dbs[session_id] = db
    logger.info("Created new unique session DB: %s", info.db_path)

    _trace_session_json(db, "SESSION_START_REQUEST", {
        "scenarioType": req.scenarioType,
        "language": req.language,
        "themes": req.themes,
        "era_context": req.era_context,
        "customPrompt": req.customPrompt,
        "picked_seed": (req.picked_seed[:1500] if req.picked_seed else ""),
        "setting_desc": setting_desc,
        "query_text": query_text,
    })

    _trace_session_json(db, "SESSION_START_ENV", {
        "prompt_dir": prompt_dir,
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma3:27b"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "scenario_synth_num_ctx": os.getenv("SCENARIO_SYNTH_NUM_CTX", "16384"),
        "scenario_synth_num_predict": os.getenv("SCENARIO_SYNTH_NUM_PREDICT", "4096"),
    })
    # Register investigators
    for inv in req.investigators:
        chars = inv.get("characteristics", {})
        attrs = inv.get("attributes", {})
        aid = db.upsert_actor(
            kind="PC",
            name=inv.get("name", "Unknown"),
            description=inv.get("occupation", ""),
            hp=attrs.get("HP", {}).get("current", 10),
            mp=attrs.get("MagicPoints", {}).get("current", 10),
            san=attrs.get("Sanity", {}).get("current", 50),
            stats={
                "str": chars.get("STR", 50),
                "con": chars.get("CON", 50),
                "dex": chars.get("DEX", 50),
                "int": chars.get("INT", 50),
                "pow": chars.get("POW", 50),
                "app": chars.get("APP", 50),
                "siz": chars.get("SIZ", 50),
                "edu": chars.get("EDU", 50),
            },
            notes=inv.get("background", ""),
        )
        for skill in inv.get("skills", []):
            db.set_skill(aid, skill["name"], skill["value"])
        db.log_event("SYS_INIT", {"note": f"Character {inv.get('name')} registered."})

    scenario_atoms_text = ""
    blueprint: dict = {}

    if req.scenarioType == "prebuilt" and req.picked_seed:
        logger.info("Prebuilt scenario selected — extracting blueprint from source content.")
        scenario_source_text = req.picked_seed
        extract_prompt = (
            "You are a Call of Cthulhu scenario analyst. "
            "Below is the full text of a published scenario. "
            "Extract its structure into JSON. Do NOT invent anything — only use what is in the text.\n\n"
            "SCENARIO TEXT:\n"
            f"{scenario_source_text[:6000]}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  title (str), era_and_setting (str), inciting_hook (str), core_mystery (str),\n"
            "  key_npc (str), hidden_threat (str), atmosphere_notes (str),\n"
            "  plot_twists (list[str]),\n"
            "  locations (list of {name, description, tags}),\n"
            "  npcs (list of {name, description, role, secret}),\n"
            "  clues (list of {title, content, location}),\n"
            "  plot_threads (list of {name, stakes, steps})\n"
            "JSON:"
        )
        try:
            extract_raw = get_llm(
                temperature=0.1,
                task="scenario_json",
            ).invoke(extract_prompt)
            _dbg("EXTRACT PROMPT", extract_prompt[:2000])
            _dbg("EXTRACT RAW RESPONSE", str(extract_raw))
            blueprint = extract_blueprint_json(extract_raw)
            scenario_atoms_text = (
                f"SCENARIO TITLE: {blueprint.get('title', 'Unknown')}\n"
                f"SETTING: {blueprint.get('era_and_setting', '')}\n"
                f"HOOK: {blueprint.get('inciting_hook', '')}\n"
                f"KEY NPC: {blueprint.get('key_npc', '')}\n"
                f"ATMOSPHERE: {blueprint.get('atmosphere_notes', '')}"
            )
            logger.info("Prebuilt blueprint extracted: %s", blueprint.get("title", "?"))
        except Exception as e:
            logger.warning("Prebuilt extraction failed (%s), using raw scenario text.", e)
            blueprint = {}
            scenario_atoms_text = scenario_source_text[:4000]

    elif scen_db:
        logger.info("Querying Scenario DB for starting atoms using: %r", query_text)

        def _build_scenario_search_queries(seed: str) -> list[str]:
            seed = (seed or "").strip()
            return [
                seed,
                f"{seed}\nFocus on the central uncanny phenomenon, the social context, and the first investigation scene.",
                f"{seed}\nFind scenario fragments with the same type of institutional or social anomaly.",
                f"{seed}\nPrefer clue sources, authority mismatch, procedural wrongness, hidden organizer, and escalating dread.",
            ]

        def _is_useful_atom(doc) -> bool:
            text = (doc.page_content or "").strip()
            meta = getattr(doc, "metadata", {}) or {}
            role = str(meta.get("role", "") or "").lower()
            abstraction = str(meta.get("abstraction", "") or "").lower()
            atom_type = str(meta.get("type", "") or "").lower()
            title_en = str(meta.get("title_en", "") or "").lower()
            display_name = str(meta.get("display_name", "") or "").lower()

            signal_text = " ".join(x for x in [text, abstraction, title_en, display_name] if x).strip()
            if len(signal_text) < 80:
                return False
            if text.lower().startswith("(implied from context"):
                return False
            if atom_type in {"appendix_character", "timeline"} and len(signal_text) < 140:
                return False
            if role.startswith("worldbuilding") or role.startswith("atmosphere"):
                if "anomaly" not in abstraction and "discovery" not in abstraction and "mystery" not in abstraction:
                    return False
            return True

        def _dedupe_docs(docs):
            seen = set()
            out = []
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                key = (
                    meta.get("source", ""),
                    meta.get("display_name", "") or meta.get("title_en", "") or meta.get("Header_2", ""),
                    meta.get("type", ""),
                    meta.get("role", ""),
                    (doc.page_content or "").strip()[:180],
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(doc)
            return out

        candidate_docs = []
        for q in _build_scenario_search_queries(query_text):
            candidate_docs.extend(scen_db.similarity_search(q, k=6))

        _trace_session_json(db, "SCENARIO_RETRIEVE_CANDIDATES", [
            _serialize_doc_for_trace(doc) for doc in candidate_docs[:20]
        ])
        
        scen_docs = _dedupe_docs(candidate_docs)
        scen_docs = [d for d in scen_docs if _is_useful_atom(d)]

        _trace_session_json(db, "SCENARIO_RETRIEVE_FILTERED", [
            _serialize_doc_for_trace(doc) for doc in scen_docs[:20]
        ])

        seed_terms = _seed_terms(query_text)
        setting_terms = _setting_terms(era_context)
        rescored = []

        for doc in scen_docs:
            text = (doc.page_content or "").strip().lower()
            meta = getattr(doc, "metadata", {}) or {}
            abstraction = str(meta.get("abstraction", "") or "").lower()
            role = str(meta.get("role", "") or "").lower()
            header = str(meta.get("Header_2", "") or "").lower()
            atom_type = str(meta.get("type", "") or "").lower()
            archetype = str(meta.get("archetype", "") or "").lower()
            title_en = str(meta.get("title_en", "") or "").lower()
            display_name = str(meta.get("display_name", "") or "").lower()
            aliases = str(meta.get("aliases", "") or "").lower()

            score = 0

            for term in seed_terms:
                if term in text:
                    score += 3
                if term in abstraction:
                    score += 3
                if term in header:
                    score += 1
                if term in title_en:
                    score += 4
                if term in display_name:
                    score += 4
                if term in aliases:
                    score += 2
                if term in archetype:
                    score += 1
                if term in atom_type:
                    score += 1

            for term in setting_terms:
                if term in text:
                    score += 4
                if term in abstraction:
                    score += 4
                if term in header:
                    score += 2
                if term in title_en or term in display_name:
                    score += 2

            score += _doc_setting_coherence(doc, era_context, query_text)

            if atom_type in ("clue", "event", "npc", "location"):
                score += 2
            if atom_type in ("appendix_character", "timeline"):
                score -= 2

            if any(x in role for x in ("clue", "investigation", "information", "mystery", "info_source", "clue_source", "objective", "setting", "context")):
                score += 2
            if any(x in role for x in ("endgame", "boss weakness", "plot resolution", "climax", "combat")):
                score -= 5
            if any(x in header for x in ("ritual", "banishment", "exorcism", "purification")):
                score -= 4

            score += min(len(text) // 300, 2)
            rescored.append((score, doc))

        rescored.sort(key=lambda x: x[0], reverse=True)

        _trace_session_json(db, "SCENARIO_RETRIEVE_RESCORED_TOP", [
            {
                "score": score,
                "doc": _serialize_doc_for_trace(doc),
            }
            for score, doc in rescored[:12]
        ])

        mixed_docs = _select_coherent_docs(rescored, target_k=4)

        _trace_session_json(db, "SCENARIO_RETRIEVE_SELECTED", [
            _serialize_doc_for_trace(doc) for doc in mixed_docs
        ])

        atoms = [f"ATOM {i + 1}:\n{_condense_atom(doc)}" for i, doc in enumerate(mixed_docs)]
        raw_atoms_text = "\n\n".join(atoms)

        _trace_session(db, "SCENARIO_RAW_ATOMS_TEXT", raw_atoms_text)
        
        _dbg("RANDOM/CUSTOM RAW ATOMS", raw_atoms_text[:12000])

        logger.info("Synthesizing unique scenario blueprint from atoms (planner -> compose)...")
        try:
            blueprint, scenario_atoms_text = _run_multi_call_scenario_synth(
                db=db,
                prompt_dir=prompt_dir,
                themes_str=themes_str,
                era_context=era_context,
                lang=lang,
                query_text=query_text,
                raw_atoms_text=raw_atoms_text,
            )
            logger.info(
                "[SCENARIO_SYNTH] title=%r | hook=%r | core=%r | hidden=%r",
                blueprint.get("title", ""),
                blueprint.get("inciting_hook", ""),
                blueprint.get("core_mystery", ""),
                blueprint.get("hidden_threat", ""),
            )
            logger.info("Scenario blueprint synthesized: %s", blueprint.get("title", "?"))
        except Exception as e:
            logger.warning("Scenario synthesis failed (%s), falling back to safe single-anchor atoms.", e)
            safe_anchor = atoms[0] if atoms else raw_atoms_text[:500]
            blueprint = {}
            scenario_atoms_text = (
                f"SCENARIO SEED: {query_text}\n"
                f"SETTING: {era_context}\n"
                f"THEMES: {themes_str}\n\n"
                f"{safe_anchor}"
            )
            _trace_session(db, "SCENARIO_SYNTH_EXCEPTION", repr(e))
            _trace_session(db, "SCENARIO_SYNTH_FALLBACK_ATOMS", scenario_atoms_text)
    else:
        scenario_atoms_text = (
            f"SCENARIO SEED: {query_text}\n"
            f"SETTING: {era_context}\n"
            f"THEMES: {themes_str}"
        )

    _dbg("FINAL scenario_atoms_text", scenario_atoms_text)
    _dbg("FINAL blueprint (first 1000)", json.dumps(blueprint, ensure_ascii=False)[:1000])
    logger.info("[SESSION] Final scenario_atoms_text (first 300): %s", scenario_atoms_text[:300])
    logger.info("[SESSION] language being stored: %r", lang)

    db.log_event("SCENARIO_GENERATED", {
        "query": query_text if req.scenarioType != "prebuilt" else blueprint.get("title", "prebuilt")
    })

    if isinstance(blueprint, dict) and blueprint:
        loc_id_map: dict[str, str] = {}

    for loc in _dict_items_only(blueprint.get("locations", [])):
        loc_name = str(loc.get("name", "Unknown") or "Unknown")
        tags_raw = loc.get("tags", "")
        tags_str = ", ".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw or "")
        lid = db.upsert_location(name=loc_name, description="", tags=tags_str)
        loc_id_map[loc_name] = lid

    for npc in _dict_items_only(blueprint.get("npcs", [])):
        role = str(npc.get("role", "neutral") or "neutral").lower()
        kind = "ENEMY" if role in ("enemy", "hidden_enemy") else "NPC"
        npc_name = str(npc.get("name", "Unknown") or "Unknown")
        npc_secret = str(npc.get("secret", "") or "")
        npc_motivation = str(npc.get("motivation", "") or "")
        notes_parts = []
        if npc_secret:
            notes_parts.append(f"SECRET: {npc_secret}")
        if npc_motivation:
            notes_parts.append(f"MOTIVATION: {npc_motivation}")

        aid = db.upsert_actor(
            kind=kind,
            name=npc_name,
            description=role,
            hp=10 if kind == "NPC" else 12,
            mp=0,
            san=50 if kind == "NPC" else 0,
            stats={
                "str": 50,
                "con": 50,
                "dex": 50,
                "int": 50,
                "pow": 50,
                "app": 40,
                "siz": 50,
                "edu": 50,
            },
            notes="\n".join(notes_parts).strip(),
        )
        db.set_skill(aid, "Dodge", 25)
        db.set_skill(aid, "Fighting (Brawl)", 25 if kind == "NPC" else 40)
        if kind == "ENEMY":
            db.set_skill(aid, "Firearms (Handgun)", 20)

    for clue in _dict_items_only(blueprint.get("clues", [])):
        loc_name = str(clue.get("location", "") or "")
        surface = str(clue.get("content", "") or "")
        deeper = str(clue.get("true_meaning", "") or "")
        stored_content = surface
        if deeper:
            stored_content = f"{surface}\nTRUE_MEANING: {deeper}" if surface else f"TRUE_MEANING: {deeper}"
        db.upsert_clue(
            title=str(clue.get("title", "Clue") or "Clue"),
            content=stored_content,
            status="hidden",
            location_id=loc_id_map.get(loc_name),
        )

    for thread in _dict_items_only(blueprint.get("plot_threads", [])):
        steps_raw = thread.get("steps", 4)
        steps = int(steps_raw) if isinstance(steps_raw, (int, float, str)) else 4
        db.upsert_thread(
            name=str(thread.get("name", "Thread") or "Thread"),
            stakes=str(thread.get("stakes", "") or ""),
            max_progress=steps,
        )

    scenario_setting = str(blueprint.get("era_and_setting") or era_context if isinstance(blueprint, dict) else era_context)

    first_act_no = 1
    first_scene_name = ""
    first_scene_location = ""
    if isinstance(blueprint, dict):
        acts = blueprint.get("acts") or []
        if acts:
            first_act = acts[0]
            first_act_no = int(first_act.get("act", 1) or 1)
            scenes = first_act.get("scenes") or []
            if scenes:
                first_scene_name = str(scenes[0].get("scene", "") or "")
                first_scene_location = str(scenes[0].get("location", "") or "")

    current_objective = _derive_initial_objective(
        blueprint if isinstance(blueprint, dict) else {},
        scenario_atoms_text,
        str(era_context),
    )

    # Keep canonical scenario data in English.
    scenario_atoms_display = scenario_atoms_text
    blueprint_display = blueprint if isinstance(blueprint, dict) else {}

    if lang != "en":
        try:
            scenario_atoms_display = translate_scenario_summary_for_user(
                scenario_atoms_text,
                lang,
            )
        except Exception as e:
            logger.warning("Scenario summary translation failed (%s); using English summary.", e)

        try:
            if isinstance(blueprint, dict) and blueprint:
                blueprint_display = translate_blueprint_for_display(blueprint, lang)
        except Exception as e:
            logger.warning("Blueprint display translation failed (%s); using English blueprint.", e)
            blueprint_display = blueprint if isinstance(blueprint, dict) else {}

    _trace_session_json(db, "SCENARIO_FINAL_STATE", {
        "scenario_atoms_text": scenario_atoms_text[:6000],
        "scenario_setting": scenario_setting,
        "era_context": era_context,
        "current_act": first_act_no,
        "current_scene": first_scene_name,
        "current_scene_location": first_scene_location,
        "current_objective": current_objective,
        "blueprint_preview": json.dumps(blueprint, ensure_ascii=False)[:6000],
    })

    cur = db.conn.cursor()
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (str(scenario_atoms_text),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (str(themes_str),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (str(era_context),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (str(req.picked_seed[:100]) if req.picked_seed else "",))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (lang,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_dir', ?)", (prompt_dir,))
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)",
        (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",),
    )
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms_display', ?)",
        (str(scenario_atoms_display),),
    )
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_display_json', ?)",
        (json.dumps(blueprint_display, ensure_ascii=False) if isinstance(blueprint_display, dict) else "{}",),
    )
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_act', ?)", (str(first_act_no),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene', ?)", (first_scene_name,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene_location', ?)", (first_scene_location,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_objective', ?)", (current_objective,))
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", str(era_context)))
    db.conn.commit()

    print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
    print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
    print(f"==>> [SESSION START] language stored        : {repr(lang)}")

    if not isinstance(blueprint, dict) or not blueprint:
        result = build_opening_fallback_result(db)
        display_result = translate_opening_result_for_user(result, lang)
        apply_state_updates(db, result)
        return display_result

    return await generate_opening_scene_logic(db, session_id=session_id)
