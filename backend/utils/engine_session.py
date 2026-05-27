import json
import os
import random
import re
import asyncio
from pathlib import Path

from langchain_core.prompts import PromptTemplate

#pylint: disable=import-error
from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest, validate_character_response_payload, validate_chat_response_payload
from utils.helpers import (
    extract_blueprint_json,
    extract_json,
    get_llm,
    _normalize_repaired_blueprint,
    read_prompt,
)
from utils.helper_story import (
    repaired_prompt,
    _seed_terms,
    _setting_terms,
    _doc_setting_coherence,
    _select_coherent_docs,
    _extract_strict_json_object,
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
    translate_chat_display_payload_for_user,
    PROMPTS_DIR
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
from utils.engine_chat import _attach_image_generation
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

# ----------------------------
# SessionSB interaction helpers
# ----------------------------

def _ingest_story_graph_from_blueprint(db: SessionDB, blueprint: dict) -> None:
    if not isinstance(blueprint, dict) or not blueprint:
        return

    db.clear_story_graph()

    loc_id_map: dict[str, str] = {}
    npc_id_map: dict[str, str] = {}
    clue_id_map: dict[str, str] = {}
    thread_id_map: dict[str, str] = {}

    def ensure_location(name: str, description: str = "", tags: str = "") -> str:
        name = (name or "").strip() or "Unknown Location"
        if name in loc_id_map:
            return loc_id_map[name]
        lid = db.upsert_location(name=name, description=description, tags=tags)
        loc_id_map[name] = lid
        return lid

    def ensure_npc(name: str, role_hint: str = "neutral") -> str:
        name = (name or "").strip()
        if not name:
            name = "Unknown NPC"
        if name in npc_id_map:
            return npc_id_map[name]

        kind = "ENEMY" if role_hint in ("enemy", "hidden_enemy") else "NPC"
        aid = db.upsert_actor(
            kind=kind,
            name=name,
            description=role_hint,
            hp=12 if kind == "ENEMY" else 10,
            mp=0,
            san=0 if kind == "ENEMY" else 50,
            stats={"str":50,"con":50,"dex":50,"int":50,"pow":50,"app":40,"siz":50,"edu":50},
            status="ok",
            notes="INGESTED_FROM_SCENE_GRAPH"
        )
        npc_id_map[name] = aid
        return aid

    def ensure_clue(title: str, content: str = "", location_id: str | None = None) -> str:
        title = (title or "").strip() or "Clue"
        if title in clue_id_map:
            return clue_id_map[title]
        cid = db.upsert_clue(
            title=title,
            content=content,
            status="hidden",
            location_id=location_id,
        )
        clue_id_map[title] = cid
        return cid

    def ensure_thread(name: str, stakes: str = "", steps: int = 3) -> str:
        name = (name or "").strip() or "Story Thread"
        if name in thread_id_map:
            return thread_id_map[name]
        tid = db.upsert_thread(name=name, progress=0, max_progress=max(steps, 1), stakes=stakes)
        thread_id_map[name] = tid
        return tid

    # Optional global registries first, but do not rely on them.
    for loc in [x for x in (blueprint.get("locations") or []) if isinstance(x, dict)]:
        ensure_location(str(loc.get("name", "") or ""),
                        tags=", ".join(loc.get("tags", [])) if isinstance(loc.get("tags"), list) else str(loc.get("tags", "") or ""))

    for npc in [x for x in (blueprint.get("npcs") or []) if isinstance(x, dict)]:
        role = str(npc.get("role", "neutral") or "neutral").lower()
        aid = ensure_npc(str(npc.get("name", "") or ""), role)
        # enrich notes if present
        notes = []
        if npc.get("secret"): notes.append(f"SECRET: {npc['secret']}")
        if npc.get("motivation"): notes.append(f"MOTIVATION: {npc['motivation']}")
        if notes:
            db.patch_actor(aid, notes="\n".join(notes))

    for th in [x for x in (blueprint.get("plot_threads") or []) if isinstance(x, dict)]:
        ensure_thread(
            str(th.get("name", "") or ""),
            stakes=str(th.get("stakes", "") or ""),
            steps=int(th.get("steps", 3) or 3),
        )

    # Exact act -> scene graph
    for act_idx, act in enumerate([x for x in (blueprint.get("acts") or []) if isinstance(x, dict)], start=1):
        act_no = int(act.get("act", act_idx) or act_idx)
        act_title = str(act.get("title", "") or f"Act {act_no}")
        act_id = db.upsert_story_act(
            act_no=act_no,
            title=act_title,
            summary=str(act.get("summary", "") or ""),
            purpose=str(act.get("purpose", "") or ""),
            belief_shift=str(act.get("belief_shift", "") or ""),
            required_payoffs=[str(x) for x in (act.get("required_payoffs") or []) if x],
            module_type=str(act.get("module_type", "") or ""),
            payload=act,
        )

        thread_id = ensure_thread(
            act_title,
            stakes=str(act.get("summary", "") or ""),
            steps=max(len(act.get("scenes") or []), 1),
        )

        for scene_idx, scene in enumerate([x for x in (act.get("scenes") or []) if isinstance(x, dict)], start=1):
            location_name = str(scene.get("location", "") or f"{act_title} Location")
            location_id = ensure_location(
                location_name,
                description=str(scene.get("description", "") or ""),
                tags=str(scene.get("scene_function", "") or ""),
            )

            scene_id = db.upsert_story_scene(
                act_id=act_id,
                act_no=act_no,
                scene_no=scene_idx,
                name=str(scene.get("scene", "") or f"Scene {scene_idx}"),
                location_id=location_id,
                payload=scene,
            )

            db.link_story_scene_thread(scene_id, thread_id)

            for objective_type, key in (
                ("trigger", "trigger"),
                ("dramatic_question", "dramatic_question"),
                ("exit_condition", "exit_condition"),
            ):
                text = str(scene.get(key, "") or "").strip()
                if text:
                    db.add_story_scene_objective(scene_id, objective_type, text)

            for key in ("npc_present", "npcs_present", "present_npcs", "scene_npcs"):
                for npc_name in [str(x) for x in (scene.get(key) or []) if x]:
                    aid = ensure_npc(npc_name)
                    db.link_story_scene_npc(scene_id, aid)

            scene_reveal_seed = "\n".join([str(x) for x in (scene.get("reveals") or [])[:2] if x]).strip()
            if not scene_reveal_seed:
                scene_reveal_seed = str(scene.get("what_happens", "") or "")

            for clue_title in [str(x) for x in (scene.get("clues_available") or []) if x]:
                cid = ensure_clue(clue_title, content=scene_reveal_seed[:500], location_id=location_id)
                db.link_story_scene_clue(scene_id, cid)

    first_scene = db.get_current_story_scene()
    if not first_scene:
        cur = db.conn.cursor()
        cur.execute("""
            SELECT s.* FROM story_scenes s
            ORDER BY s.act_no, s.scene_no LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            db.set_current_story_scene(row["id"])



# ----------------------------
# Prompt prep / LLM helpers
# ----------------------------

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

def _looks_like_prebuilt_atoms(text: str) -> bool:
    t = str(text or "")
    return (
        "# SOURCE:" in t
        or "TITLE_EN:" in t
        or "DISPLAY_NAME:" in t
        or "TYPE:" in t
        or "ARCHETYPE:" in t
    )


def _extract_prebuilt_source_name(text: str) -> str:
    m = re.search(r"^#\s*SOURCE:\s*(.+)$", str(text or ""), re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def _extract_prebuilt_title_guess(text: str) -> str:
    t = str(text or "")

    m = re.search(r"^##\s+(.+)$", t, re.MULTILINE)
    if m:
        return m.group(1).strip()

    src = _extract_prebuilt_source_name(t)
    if src:
        stem = Path(src).stem.replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", stem).strip()

    return "Prebuilt Scenario"

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

def _normalize_blueprint_registries(blueprint: dict) -> dict:
    bp = dict(blueprint or {})
    acts = bp.get("acts") or []
    if not isinstance(acts, list):
        return bp

    locations = bp.get("locations")
    npcs = bp.get("npcs")
    clues = bp.get("clues")
    plot_threads = bp.get("plot_threads")

    if not isinstance(locations, list):
        locations = []
    if not isinstance(npcs, list):
        npcs = []
    if not isinstance(clues, list):
        clues = []
    if not isinstance(plot_threads, list):
        plot_threads = []

    seen_loc = {str(x.get("name", "")).strip().lower() for x in locations if isinstance(x, dict)}
    seen_npc = {str(x.get("name", "")).strip().lower() for x in npcs if isinstance(x, dict)}
    seen_clue = {str(x.get("title", "")).strip().lower() for x in clues if isinstance(x, dict)}
    seen_thread = {str(x.get("name", "")).strip().lower() for x in plot_threads if isinstance(x, dict)}

    for act in acts:
        scenes = act.get("scenes") or []
        if not isinstance(scenes, list):
            continue

        act_title = str(act.get("title", "") or "").strip()
        if act_title and act_title.lower() not in seen_thread:
            plot_threads.append({
                "name": act_title,
                "stakes": str(act.get("summary", "") or "").strip(),
                "steps": max(3, len(scenes) or 3),
            })
            seen_thread.add(act_title.lower())

        for scene in scenes:
            if not isinstance(scene, dict):
                continue

            location = str(scene.get("location", "") or "").strip()
            description = str(scene.get("description", "") or "").strip()
            what_happens = str(scene.get("what_happens", "") or "").strip()
            dramatic_question = str(scene.get("dramatic_question", "") or "").strip()
            scene_name = str(scene.get("scene", "") or "").strip()

            scene.setdefault("reveals", [])
            scene.setdefault("conceals", [])
            scene.setdefault("clues_available", [])
            scene.setdefault("npc_present", [])

            if location and location.lower() not in seen_loc:
                locations.append({
                    "name": location,
                    "tags": str(scene.get("scene_function", "") or "").strip(),
                    "hidden": dramatic_question or what_happens[:140],
                })
                seen_loc.add(location.lower())

            clue_candidates = []

            if dramatic_question:
                clue_candidates.append(dramatic_question)
            if what_happens:
                clue_candidates.append(what_happens)
            if description:
                clue_candidates.append(description)

            for idx, clue_text in enumerate(clue_candidates, start=1):
                clue_title = f"{scene_name} clue {idx}".strip()
                if clue_title.lower() in seen_clue:
                    continue

                clues.append({
                    "title": clue_title,
                    "content": clue_text[:220],
                    "true_meaning": "",
                    "location": location,
                })
                scene["clues_available"].append(clue_title)
                seen_clue.add(clue_title.lower())

            reveal_text = what_happens or dramatic_question
            if reveal_text and reveal_text not in scene["reveals"]:
                scene["reveals"].append(reveal_text)

    bp["acts"] = acts
    bp["locations"] = locations
    bp["npcs"] = npcs
    bp["clues"] = clues
    bp["plot_threads"] = plot_threads
    return bp

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
        task="scenario_plan",
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
                    task="scenario_compose",
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
                    task="scenario_compose",
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
            task="scenario_repair",
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
    lang = normalize_language_code(req.language)

    llm = get_llm(
        temperature=0.7,
        task="character_json",
    )
    chain = PromptTemplate.from_template(
        read_prompt("character_gen.txt", prompt_dir=PROMPTS_DIR)
    ) | llm

    response_text = chain.invoke({
        "language": lang,
        "language_name": get_language_name(lang),
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected",
    })

    raw = str(response_text or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
    raw = re.sub(r"\n```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")

    parsed = {}
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            parsed = json.loads(candidate, strict=False)
        except Exception as e:
            logger.warning("Character generation JSON parse failed: %s", e)
            logger.warning("Character generation raw candidate: %r", candidate[:4000])

    if not isinstance(parsed, dict):
        parsed = {}

    return validate_character_response_payload(parsed)


# ----------------------------
# Opening scene
# ----------------------------

def _current_opening_scene_labels(db: SessionDB) -> tuple[str, str]:
    """
    Prefer the DB story graph over legacy kv_store keys.
    """
    scene = db.get_current_story_scene()
    if scene:
        return (
            str(scene.get("name", "") or "").strip(),
            str(scene.get("location_name", "") or "").strip(),
        )

    row_scene = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='current_scene'"
    ).fetchone()
    row_loc = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='current_scene_location'"
    ).fetchone()

    return (
        str(row_scene[0] if row_scene else "" or "").strip(),
        str(row_loc[0] if row_loc else "" or "").strip(),
    )


def _force_opening_scene_entities_from_db(db: SessionDB, result: dict) -> dict:
    """
    Opening scene entities must come from current DB story scene only.
    Do not trust LLM-invented names.
    """
    safe = dict(result or {})
    scene = db.get_current_story_scene()

    names: list[str] = []
    if scene:
        names = [
            str(npc.get("name", "") or "").strip()
            for npc in db.list_story_scene_npcs(scene["id"])
            if str(npc.get("name", "") or "").strip()
        ]

    llm_entities = ((safe.get("scene_entities") or {}).get("present_named_entities") or [])
    dropped = [
        str(name).strip()
        for name in llm_entities
        if str(name).strip() and str(name).strip() not in names
    ]
    if dropped:
        logger.warning("OPENING_DROPPED_UNKNOWN_SCENE_ENTITIES: %s", dropped)

    safe["scene_entities"] = {"present_named_entities": names}
    return safe


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

    action_map = {
        shown: canonical
        for shown, canonical in zip(display_actions, canonical_actions)
        if shown and canonical
    }

    db.conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('last_suggested_action_map_json', ?)",
        (json.dumps(action_map, ensure_ascii=False),),
    )
    db.conn.commit()


def _finalize_opening_result(
    *,
    db: SessionDB,
    session_id: str,
    result: dict,
    session_language: str,
    setting_override: str,
    era_override: str,
) -> dict:
    """
    Single post-processing path for normal and fallback openings.
    Uses one batched display-translation call and deterministic image prompt creation.
    """
    result = validate_chat_response_payload(result)
    result = _force_opening_scene_entities_from_db(db, result)

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    apply_state_updates(db, result)

    display_result = translate_chat_display_payload_for_user(result, session_language)
    _store_display_to_canonical_action_map(db, result, display_result)

    logger.info(
        "Opening translation | lang=%s | narrative_before=%r | narrative_after=%r | actions_before=%r | actions_after=%r",
        session_language,
        result.get("narrative", "")[:180],
        display_result.get("narrative", "")[:180],
        (result.get("suggested_actions") or [])[:3],
        (display_result.get("suggested_actions") or [])[:3],
    )

    display_result = _attach_image_generation(
        session_id=session_id,
        display_result=display_result,
        canonical_narrative=result.get("narrative", ""),
        setting=setting_override,
        era=era_override,
    )

    db.log_event("SYS_OPENING_SCENE", {"content": "Opening scene generated"})
    db.log_event("CHAT", {
        "role": "Keeper",
        "content": result.get("narrative", ""),
        "display_content": display_result.get("narrative", ""),
    })

    cur = db.conn.cursor()
    row_tc = cur.execute("SELECT value FROM kv_store WHERE key='turn_count'").fetchone()
    turn_count = int(row_tc[0]) + 1 if row_tc else 1
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)",
        (str(turn_count),),
    )
    db.conn.commit()

    return display_result

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

    current_scene_name, current_scene_location = _current_opening_scene_labels(db)

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
            f"CURRENT SCENE NAME (use this exact scene): {current_scene_name}\n"
            f"CURRENT SCENE LOCATION (reuse this exact location label in state_updates.location_name): {current_scene_location}\n"
            "Do not skip ahead.\n"
            "Do not move to another scene or another location.\n"
            "Do not invent named NPCs unless they are listed in AUTHORITATIVE CURRENT SCENE / SCENE NPCS.\n"
            "scene_entities.present_named_entities must contain only names already listed in CURRENT GAME STATE or SCENE NPCS.\n"
            "state_updates.location_name must match the current scene location exactly.\n"
            "Do not require a roll unless immediate uncertainty is already established.\n"
            "Do not set clue_found or thread_progress in the opening unless investigators already learned actionable content.\n"
            "Provide exactly three actionable suggested actions."
        ),
        "last_turn_ban": "",
    })

    logger.info("generate_opening_scene_logic(): raw LLM response preview: %r", str(response_text)[:4000])
    result = extract_json(response_text)

    opening_violations = validate_opening_scene_response(db, result)
    if opening_violations:
        logger.warning("Opening scene validation warnings: %s", opening_violations)
        db.log_event("OPENING_VALIDATION_WARN", {
            "violations": opening_violations,
            "narrative": result.get("narrative", "")[:500],
        })

        hard_opening_prefixes = (
            "opening_missing_narrative",
            "opening_missing_actions",
            "opening_hidden_threat",
            "opening_final_truth",
        )

        hard_opening = any(
            v == "opening_missing_narrative"
            or v == "opening_missing_actions"
            or v.startswith("opening_hidden_threat")
            or v.startswith("opening_final_truth")
            for v in opening_violations
        )

        if hard_opening:
            logger.warning("Opening scene hard validation failed, using fallback: %s", opening_violations)
            result = build_opening_fallback_result(db)

    return _finalize_opening_result(
        db=db,
        session_id=session_id,
        result=result,
        session_language=session_language,
        setting_override=setting_override,
        era_override=era_override,
    )

# ----------------------------
# Session start
# ----------------------------

async def start_session_logic(req: StartSessionRequest) -> dict:
    session_id = "local_session"
    lang = normalize_language_code(req.language)

    themes_str = ", ".join(req.themes).upper() if req.themes else "STANDARD"
    era_context = req.era_context or "Cosmic Horror — derive era and aesthetics from the scenario atoms."

    if req.scenarioType == "custom" and req.customPrompt:
        setting_desc = f"Custom: {req.customPrompt[:50]}..."
        query_text = req.customPrompt
    elif req.scenarioType == "prebuilt" and req.picked_seed:
        setting_desc = "Prebuilt scenario"
        query_text = "Prebuilt Scenario"
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
        "prompt_dir": PROMPTS_DIR,
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
        for item in inv.get("inventory", []):
            db.add_actor_item(aid, str(item))
        db.log_event("SYS_INIT", {"note": f"Character {inv.get('name')} registered."})

    scenario_atoms_text = ""
    blueprint: dict = {}

    if req.scenarioType == "prebuilt" and req.picked_seed:
        prebuilt_payload = str(req.picked_seed or "").strip()

        if _looks_like_prebuilt_atoms(prebuilt_payload):
            logger.info("Prebuilt scenario selected — synthesizing blueprint from stored scenario atoms.")

            source_name = _extract_prebuilt_source_name(prebuilt_payload)
            title_guess = _extract_prebuilt_title_guess(prebuilt_payload)

            # Keep prebuilt sessions anchored to the supplied scenario atoms.
            query_text = title_guess or source_name or "Prebuilt Scenario"
            era_context = req.era_context or (
                "Derive era and setting strictly from the prebuilt scenario atoms. "
                "Do not invent or substitute a different era."
            )

            raw_atoms_text = prebuilt_payload

            try:
                blueprint, scenario_atoms_text = _run_multi_call_scenario_synth(
                    db=db,
                    prompt_dir=PROMPTS_DIR,
                    themes_str=themes_str,
                    era_context=era_context,
                    lang=lang,
                    query_text=query_text,
                    raw_atoms_text=raw_atoms_text,
                )
                logger.info("Prebuilt blueprint synthesized from atoms: %s", blueprint.get("title", "?"))
            except Exception as e:
                logger.warning("Prebuilt atom synthesis failed (%s), using raw prebuilt atoms.", e)
                blueprint = {}
                scenario_atoms_text = raw_atoms_text[:4000]
                _trace_session(db, "PREBUILT_SYNTH_EXCEPTION", repr(e))
                _trace_session(db, "PREBUILT_SYNTH_FALLBACK_ATOMS", scenario_atoms_text)

        else:
            logger.info("Prebuilt scenario selected — payload looks like raw scenario text, using it as atoms fallback.")
            blueprint = {}
            scenario_atoms_text = prebuilt_payload[:4000]
            era_context = req.era_context or (
                "Derive era and setting strictly from the prebuilt scenario text. "
                "Do not invent or substitute a different era."
            )

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
                prompt_dir=PROMPTS_DIR,
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
        _ingest_story_graph_from_blueprint(db, blueprint)
    
    cur = db.conn.cursor()
    first_scene = db.get_current_story_scene()
    if first_scene:
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_act', ?)", (str(first_scene["act_no"]),))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene', ?)", (str(first_scene["name"]),))
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene_location', ?)", (str(first_scene.get("location_name", "") or ""),))

    scenario_setting = str(
        blueprint.get("era_and_setting") or era_context
        if isinstance(blueprint, dict) else era_context
    )

    first_act_no = 1
    first_scene_name = ""
    first_scene_location = ""

    first_scene = db.get_current_story_scene()
    if first_scene and first_scene.get("location_id"):
        for pc in db.list_actors("PC"):
            db.patch_actor(actor_id=pc["id"], location_id=first_scene["location_id"])
            
    if first_scene:
        first_act_no = int(first_scene["act_no"])
        first_scene_name = str(first_scene["name"] or "")
        first_scene_location = str(first_scene.get("location_name", "") or "")

    scene_objective = ""
    if first_scene:
        scene_objective = db.get_story_scene_primary_objective(first_scene["id"])

    current_objective = scene_objective or _derive_initial_objective(
        blueprint if isinstance(blueprint, dict) else {},
        scenario_atoms_text,
        str(era_context),
    )

    # Keep canonical scenario data in English.
    # Do not translate large scenario summaries/blueprints during session start;
    # it is expensive and not needed for runtime play.
    scenario_atoms_display = scenario_atoms_text
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

    # if isinstance(blueprint, dict) and blueprint:
    #     blueprint = _normalize_blueprint_registries(blueprint)

  
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (str(scenario_atoms_text),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (str(themes_str),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (str(era_context),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (str(req.picked_seed[:100]) if req.picked_seed else "",))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (lang,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_dir', ?)", (str(PROMPTS_DIR),))

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
    # cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_act', ?)", (str(first_act_no),))
    # cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene', ?)", (first_scene_name,))
    # cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene_location', ?)", (first_scene_location,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_objective', ?)", (current_objective,))
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", str(era_context)))
    db.conn.commit()

    print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
    print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
    print(f"==>> [SESSION START] language stored        : {repr(lang)}")

    if not isinstance(blueprint, dict) or not blueprint:
        result = build_opening_fallback_result(db)
        return _finalize_opening_result(
            db=db,
            session_id=session_id,
            result=result,
            session_language=lang,
            setting_override=scenario_setting,
            era_override=str(era_context),
        )

    return await generate_opening_scene_logic(db, session_id=session_id)
