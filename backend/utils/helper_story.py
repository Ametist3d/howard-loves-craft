import json
import re
import logging
from pathlib import Path
#pylint: disable=import-error
from utils.db_session import SessionDB
from utils.helpers import get_llm, kv_get, detect_degenerate_output, clean_degenerate_value
from utils.prompt_translate import get_language_name
from langchain_core.prompts import PromptTemplate
from img_gen.comfy_client import BASE_URL as _COMFY_BASE_URL

logger = logging.getLogger("keeper_ai.helpers.story")


local_composed_template = """
    Generate STRICT JSON only.
    Keep every string under 18 words.
    No repeated words. No repeated names.

    Use this exact minimal schema first:
    {
    "title": "",
    "era_and_setting": "",
    "atmosphere_notes": "",
    "inciting_hook": "",
    "core_mystery": "",
    "hidden_threat": "",
    "truth_the_players_never_suspect": "",
    "acts": [
        {
        "act": 1,
        "title": "",
        "summary": "",
        "purpose": "",
        "belief_shift": "",
        "required_payoffs": ["", ""],
        "scenes": []
        }
    ]
    }

    Acts must be exactly __TARGET_ACT_COUNT__.
    Per-act scene counts must be exactly __SCENE_COUNT_PLAN__.

    SEED:
    __SEED__

    SETTING:
    __ERA_CONTEXT__

    ATOMS:
    __ATOMS__

    PLAN:
    __PLAN_JSON__
    """

repaired_prompt = """
    Repair the broken scenario blueprint JSON.

    Return STRICT JSON only.
    Do not explain anything.
    Do not invent extra wrapper keys.

    The repaired JSON MUST contain these required top-level keys:
    - title
    - era_and_setting
    - atmosphere_notes
    - inciting_hook
    - core_mystery
    - hidden_threat
    - truth_the_players_never_suspect
    - scenario_engine
    - acts
    - locations
    - npcs
    - clues
    - plot_threads

    Acts must be exactly __ACT_COUNT__.
    Per-act scene counts must be exactly __SCENE_COUNTS__.

    If a field is damaged or missing, reconstruct it briefly from the surviving context.
    If unsure, use short conservative text.
    Remove repeated phrases and repeated tokens.
    Keep all string values short.

    Use this exact structural skeleton:
    {{
    "title": "",
    "era_and_setting": "",
    "atmosphere_notes": "",
    "inciting_hook": "",
    "core_mystery": "",
    "hidden_threat": "",
    "truth_the_players_never_suspect": "",
    "scenario_engine": {{
        "surface_explanation": "",
        "actual_explanation": "",
        "false_leads": ["", ""],
        "contradictions": ["", ""],
        "reversals": ["", ""],
        "dynamic_pressures": ["", ""],
        "climax_choices": [
        {{"option": "", "cost": "", "consequence": ""}},
        {{"option": "", "cost": "", "consequence": ""}}
        ]
    }},
    "acts": [],
    "locations": [],
    "npcs": [],
    "clues": [],
    "plot_threads": []
    }}

    BROKEN JSON:
    __COMPOSE_RAW__
    """


_DOC_TERM_STOP = {
    "the", "and", "with", "from", "that", "this", "have", "were", "their",
    "into", "through", "about", "there", "which", "than", "then", "they",
    "them", "those", "these", "where", "when", "what", "using", "used",
    "scene", "event", "clue", "context", "background", "role", "type",
    "display", "name", "title", "original", "aliases"
}

def _invoke_with_retry(
    prompt: str,
    *,
    task: str = "scenario_tagged",
    num_ctx: int = 8192,
    num_predict: int = 900,
    max_attempts: int = 2,
) -> str:
    last_raw = ""

    # for Gemma4 no-json translation/plain-text tasks, leave headroom;
    # for Gemma3 tagged generation these values are also safe
    num_predict = max(num_predict, 1400)

    for attempt in range(1, max_attempts + 1):
        temp = 0.45 + (attempt - 1) * 0.10
        raw = str(get_llm(
            temperature=temp,
            task=task,
            num_ctx=num_ctx,
            num_predict=num_predict,
        ).invoke(prompt))

        last_raw = raw
        logger.info(
            "OLLAMA CALL task=%s attempt=%s temp=%.2f raw_len=%s",
            task, attempt, temp, len(raw or "")
        )

        if raw and raw.strip() and not detect_degenerate_output(raw):
            return raw

        logger.warning("Empty/degenerate output attempt %d/%d for task=%s", attempt, max_attempts, task)
        num_predict = min(num_predict + 400, 3200)

    return last_raw

def _extract_strict_json_object(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
    raw = re.sub(r"\n```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No complete JSON object found")
    return json.loads(raw[start:end + 1], strict=False)

def _build_deterministic_opening_result(db: SessionDB) -> dict:
    cur = db.conn.cursor()

    scenario_setting = kv_get(cur, "scenario_setting", "")
    scenario_title = ""
    blueprint_raw = kv_get(cur, "scenario_blueprint_json", "{}")
    try:
        blueprint = json.loads(blueprint_raw)
    except Exception:
        blueprint = {}

    if isinstance(blueprint, dict):
        scenario_title = str(blueprint.get("title", "") or "")

    current_scene = kv_get(cur, "current_scene", "Scene 1")
    current_scene_location = kv_get(cur, "current_scene_location", "the current location")
    current_objective = kv_get(cur, "current_objective", "Follow the current lead.")

    hook = ""
    if isinstance(blueprint, dict):
        hook = str(blueprint.get("inciting_hook", "") or "")

    narrative = (
        f"{scenario_title or 'The case'} begins in {current_scene_location}. "
        f"The investigators enter {current_scene}. "
        f"Something is wrong here already: {hook or 'the ordinary pattern no longer holds'}. "
        f"Right now, the priority is clear: {current_objective}"
    )

    actions = [
        f"Examine the immediate anomaly in {current_scene_location}",
        "Question the nearest witness or resident about what they consider normal here",
        "Search for physical evidence that proves the pattern is repeating",
    ]

    return {
        "narrative": narrative,
        "suggested_actions": actions,
        "roll_request": {"required": False, "skill_name": "", "action_text": "", "reason": ""},
        "state_updates": {
            "character_name": "",
            "hp_change": 0,
            "sanity_change": 0,
            "mp_change": 0,
            "inventory_add": "",
            "inventory_remove": "",
            "location_name": current_scene_location,
            "location_description": scenario_setting[:220],
            "clue_found": "",
            "clue_content": "",
            "thread_progress": "The investigators are now actively engaging the first anomaly.",
        },
    }

def _build_blueprint_scaffold(
    *,
    act_count: int,
    scene_counts: list[int],
    era_context: str,
    seed: str,
    plan: dict,
) -> dict:
    blueprint = {
        "title": "",
        "era_and_setting": era_context,
        "atmosphere_notes": "",
        "inciting_hook": seed,
        "core_mystery": "",
        "hidden_threat": "",
        "truth_the_players_never_suspect": "",
        "scenario_engine": {
            "surface_explanation": "A plausible surface explanation seems true at first.",
            "actual_explanation": "A hidden mechanism is driving the events behind the surface anomaly.",
            "false_leads": plan.get("false_leads", [])[:2],
            "contradictions": plan.get("contradictions", [])[:2],
            "reversals": plan.get("reversals", [])[:2],
            "dynamic_pressures": plan.get("dynamic_pressures", [])[:2],
            "climax_choices": [
                {"option": "", "cost": "", "consequence": ""},
                {"option": "", "cost": "", "consequence": ""},
            ],
        },
        "acts": [],
        "locations": [],
        "npcs": [],
        "clues": [],
        "plot_threads": [],
    }

    module_types = plan.get("act_module_types", [])
    for i in range(act_count):
        act_no = i + 1
        module_type = module_types[i] if i < len(module_types) else "surface_inquiry"
        act = {
            "act": act_no,
            "title": "",
            "summary": "",
            "purpose": "",
            "belief_shift": "",
            "required_payoffs": ["", ""],
            "module_type": module_type,
            "scenes": [],
        }

        for j in range(scene_counts[i]):
            scene_no = j + 1
            act["scenes"].append({
                "scene": f"Scene {scene_no}",
                "location": "",
                "scene_function": "investigation",
                "dramatic_question": "",
                "entry_condition": "",
                "exit_condition": "",
                "trigger": "",
                "description": "",
                "what_happens": "",
                "pressure_if_delayed": "",
                "reveals": [],
                "conceals": [],
                "clues_available": [],
                "npc_present": [],
                "threat_level": "tension",
                "keeper_notes": "",
            })

        blueprint["acts"].append(act)

    return blueprint

def _canon_tag(key: str) -> str:
    key = key.strip().upper()
    key = key.replace("-", "_").replace(" ", "_")
    key = re.sub(r"__+", "_", key)
    return key

def _extract_tagged_fields(text: str, allowed_keys: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    raw = (text or "").strip()

    canon_allowed = {_canon_tag(k) for k in allowed_keys}

    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = _canon_tag(key)
        value = clean_degenerate_value(value.strip())
        if not value:
            continue

        if key in canon_allowed:
            out[key] = value

    return out


def _generate_blueprint_header(
    *,
    era_context: str,
    seed: str,
    atoms: str,
    plan: dict,
) -> dict:
    prompt = f"""
Write rich scenario header fields.

OUTPUT RULES:
- English only
- One field per line
- Use EXACT tags below
- No JSON
- No markdown
- TITLE under 8 words
- ERA_AND_SETTING: 1 sentence
- ATMOSPHERE_NOTES: 1 sentence
- INCITING_HOOK: 1-2 sentences
- CORE_MYSTERY: 2-3 sentences
- HIDDEN_THREAT: 2-4 sentences
- TRUTH_THE_PLAYERS_NEVER_SUSPECT: 2-3 sentences
- SURFACE_EXPLANATION: 1-2 sentence
- ACTUAL_EXPLANATION: 2-3 sentences
- CLIMAX choices may be 2-3 sentences each
- Do not repeat phrases

TAGS:
TITLE:
ERA_AND_SETTING:
ATMOSPHERE_NOTES:
INCITING_HOOK:
CORE_MYSTERY:
HIDDEN_THREAT:
TRUTH_THE_PLAYERS_NEVER_SUSPECT:
SURFACE_EXPLANATION:
ACTUAL_EXPLANATION:
CLIMAX_1_OPTION:
CLIMAX_1_COST:
CLIMAX_1_CONSEQUENCE:
CLIMAX_2_OPTION:
CLIMAX_2_COST:
CLIMAX_2_CONSEQUENCE:

SEED:
{seed}

SETTING:
{era_context}

ATOMS:
{atoms}

PLAN:
{json.dumps(plan, ensure_ascii=False)}
"""
    raw = _invoke_with_retry(
        prompt,
        task="scenario_tagged",
        num_ctx=8192,
        num_predict=1400,
    )
    logger.info("SCENARIO_HEADER_RAW: %r", str(raw)[:6000])

    allowed = {
        "TITLE", "ERA_AND_SETTING", "ATMOSPHERE_NOTES", "INCITING_HOOK",
        "CORE_MYSTERY", "HIDDEN_THREAT", "TRUTH_THE_PLAYERS_NEVER_SUSPECT",
        "SURFACE_EXPLANATION", "ACTUAL_EXPLANATION",
        "CLIMAX_1_OPTION", "CLIMAX_1_COST", "CLIMAX_1_CONSEQUENCE",
        "CLIMAX_2_OPTION", "CLIMAX_2_COST", "CLIMAX_2_CONSEQUENCE",
    }
    parsed = _extract_tagged_fields(raw, allowed)

    # fill gaps from seed/plan so caller never gets empty strings
    seed_words = [w for w in seed.split() if w.lower() not in ("a", "an", "the")]
    short_seed = " ".join(seed.split()[:15]) if seed else "an unknown disturbance"

    noun = seed_words[0].title() if seed_words else "Unknown"
    defaults = {
        "TITLE": f"The {noun} Incident",
        "ERA_AND_SETTING": era_context,
        "ATMOSPHERE_NOTES": "Tense, uncanny, and escalating.",
        "INCITING_HOOK": short_seed,
        "CORE_MYSTERY": f"The surface event conceals a deeper mechanism connected to: {short_seed}",
        "HIDDEN_THREAT": "A concealed predatory force exploiting the environment and human vulnerability.",
        "TRUTH_THE_PLAYERS_NEVER_SUSPECT": "The anomaly is active, strategic, and not merely environmental.",
        "SURFACE_EXPLANATION": plan.get("false_leads", ["A plausible mundane explanation"])[0],
        "ACTUAL_EXPLANATION": "A hidden mechanism drives events behind the surface anomaly.",
        "CLIMAX_1_OPTION": "Contain the threat at personal cost",
        "CLIMAX_1_COST": "Immediate personal loss",
        "CLIMAX_1_CONSEQUENCE": "The immediate danger is reduced but the cost remains.",
        "CLIMAX_2_OPTION": "Expose the truth publicly",
        "CLIMAX_2_COST": "Escalation and retaliation",
        "CLIMAX_2_CONSEQUENCE": "The hidden pattern becomes visible but fallout spreads.",
    }

    for key, fallback in defaults.items():
        if key not in parsed or not parsed[key]:
            parsed[key] = fallback

    return parsed

_MODULE_VERBS = {
    "surface_inquiry": ("investigate", "question witnesses", "compare accounts"),
    "danger_probe": ("enter the restricted area", "confront the hazard", "push into danger"),
    "contradiction_reveal": ("uncover the contradiction", "disprove the surface theory", "expose the inconsistency"),
    "costly_climax": ("make the final choice", "confront the source", "pay the price"),
    "witness_network": ("cross-reference testimonies", "pressure a reluctant witness", "map the social web"),
    "site_investigation": ("search the site", "examine physical evidence", "map the location"),
    "institutional_obstruction": ("bypass authority", "negotiate access", "find another way in"),
    "parallel_leads": ("split the investigation", "follow the diverging lead", "compare both threads"),
    "false_pattern_strengthens": ("follow the false lead deeper", "commit to the wrong theory", "gather misleading evidence"),
    "hidden_logistics": ("trace the supply chain", "find the operational pattern", "expose the hidden route"),
    "social_crack": ("pressure the weakest link", "exploit the NPC's doubt", "force a confession"),
    "betrayal_or_complicity": ("discover the betrayal", "confront the accomplice", "reveal hidden loyalty"),
    "reversal_of_cause": ("reinterpret the evidence", "realize the cause is an effect", "reverse the assumption"),
    "negotiated_window": ("attempt negotiation", "buy time", "propose terms"),
    "countdown_crisis": ("race against the deadline", "act before it's too late", "make a rushed decision"),
    "chase_or_containment": ("pursue the threat", "block the escape", "contain the breach"),
    "converging_threads": ("connect the threads", "see the full picture", "unify the evidence"),
}


def _generate_act_payload(
    *,
    act_no: int,
    module_type: str,
    scene_count: int,
    seed: str,
    era_context: str,
    atoms: str,
    plan: dict,
    previous_acts_summary: str,
) -> dict:
    prompt = f"""
Write one act payload.

OUTPUT RULES:
- English only
- No JSON
- No markdown
- One field per line
- Keep values short
- Do not repeat phrases
- Produce exactly {scene_count} scenes

ACT TAGS:
ACT_TITLE:
ACT_SUMMARY:
ACT_PURPOSE:
ACT_BELIEF_SHIFT:
ACT_REQUIRED_PAYOFF_1:
ACT_REQUIRED_PAYOFF_2:

For each scene, use:
SCENE_1_NAME:
SCENE_1_LOCATION:
SCENE_1_FUNCTION:
SCENE_1_DRAMATIC_QUESTION:
SCENE_1_ENTRY:
SCENE_1_EXIT:
SCENE_1_TRIGGER:
SCENE_1_DESCRIPTION:
SCENE_1_WHAT_HAPPENS:
SCENE_1_PRESSURE_IF_DELAYED:
SCENE_1_THREAT_LEVEL:
SCENE_1_KEEPER_NOTES:

Repeat the same pattern up to SCENE_{scene_count}_...

Act number: {act_no}
Module type: {module_type}

SEED:
{seed}

SETTING:
{era_context}

ATOMS:
{atoms}

PLAN:
{json.dumps(plan, ensure_ascii=False)}

PREVIOUS ACTS SUMMARY:
{previous_acts_summary}
"""
    raw = _invoke_with_retry(
        prompt,
        task="scenario_tagged",
        num_ctx=9000,
        num_predict=1800,
    )
    logger.info("SCENARIO_ACT_RAW[%s]: %r", act_no, str(raw)[:8000])

    scene_suffixes = [
        "NAME", "LOCATION", "FUNCTION", "DRAMATIC_QUESTION",
        "ENTRY", "EXIT", "TRIGGER", "DESCRIPTION",
        "WHAT_HAPPENS", "PRESSURE_IF_DELAYED", "THREAT_LEVEL", "KEEPER_NOTES",
    ]
    allowed = {
        "ACT_TITLE", "ACT_SUMMARY", "ACT_PURPOSE",
        "ACT_BELIEF_SHIFT", "ACT_REQUIRED_PAYOFF_1", "ACT_REQUIRED_PAYOFF_2",
        *{f"SCENE_{i}_{s}" for i in range(1, scene_count + 1) for s in scene_suffixes},
    }
    lines = _extract_tagged_fields(raw, allowed)

    verbs = _MODULE_VERBS.get(module_type, ("investigate", "advance", "decide"))

    # build scenes with seed-aware defaults
    scenes = []
    for i in range(1, scene_count + 1):
        verb = verbs[min(i - 1, len(verbs) - 1)]
        scenes.append({
            "scene": lines.get(f"SCENE_{i}_NAME") or f"Scene {i}: {verb.title()}",
            "location": lines.get(f"SCENE_{i}_LOCATION") or "Unfixed location",
            "scene_function": lines.get(f"SCENE_{i}_FUNCTION") or "investigation",
            "dramatic_question": lines.get(f"SCENE_{i}_DRAMATIC_QUESTION") or f"Can the investigators {verb}?",
            "entry_condition": lines.get(f"SCENE_{i}_ENTRY") or "Investigators pursue the current lead.",
            "exit_condition": lines.get(f"SCENE_{i}_EXIT") or "They leave with a clearer direction.",
            "trigger": lines.get(f"SCENE_{i}_TRIGGER") or f"Evidence or pressure forces the next step.",
            "description": lines.get(f"SCENE_{i}_DESCRIPTION") or f"The investigators attempt to {verb}.",
            "what_happens": lines.get(f"SCENE_{i}_WHAT_HAPPENS") or f"The situation changes as they {verb}.",
            "pressure_if_delayed": lines.get(f"SCENE_{i}_PRESSURE_IF_DELAYED") or "The situation worsens.",
            "reveals": [],
            "conceals": [],
            "clues_available": [],
            "npc_present": [],
            "threat_level": lines.get(f"SCENE_{i}_THREAT_LEVEL") or "tension",
            "keeper_notes": lines.get(f"SCENE_{i}_KEEPER_NOTES") or "Keep the scene concise and playable.",
        })

    return {
        "title": lines.get("ACT_TITLE") or f"Act {act_no}: {module_type.replace('_', ' ').title()}",
        "summary": lines.get("ACT_SUMMARY") or f"The investigators {verbs[0]}.",
        "purpose": lines.get("ACT_PURPOSE") or f"Advance through {module_type.replace('_', ' ')}.",
        "belief_shift": lines.get("ACT_BELIEF_SHIFT") or "The initial explanation weakens.",
        "required_payoffs": [
            lines.get("ACT_REQUIRED_PAYOFF_1") or "evidence",
            lines.get("ACT_REQUIRED_PAYOFF_2") or "pressure",
        ],
        "scenes": scenes,
    }


def _normalize_act_payload(payload: dict, *, act_no: int, scene_count: int, module_type: str) -> dict:
    payload = dict(payload or {})
    payload.setdefault("title", f"Act {act_no}")
    payload.setdefault("summary", "The investigation advances.")
    payload.setdefault("purpose", "Advance the scenario.")
    payload.setdefault("belief_shift", "The initial explanation weakens.")
    payload.setdefault("required_payoffs", ["evidence", "pressure"])

    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        scenes = []

    while len(scenes) < scene_count:
        scenes.append({})

    scenes = scenes[:scene_count]

    norm_scenes = []
    for i, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            scene = {}
        norm_scenes.append({
            "scene": str(scene.get("scene") or f"Scene {i}"),
            "location": str(scene.get("location") or "Unfixed location"),
            "scene_function": str(scene.get("scene_function") or "investigation"),
            "dramatic_question": str(scene.get("dramatic_question") or "What changes here?"),
            "entry_condition": str(scene.get("entry_condition") or "Investigators pursue the current lead."),
            "exit_condition": str(scene.get("exit_condition") or "They leave with a clearer direction."),
            "trigger": str(scene.get("trigger") or "A clue or pressure forces movement."),
            "description": str(scene.get("description") or "The investigators confront a new layer of the problem."),
            "what_happens": str(scene.get("what_happens") or "Evidence and pressure reshape the situation."),
            "pressure_if_delayed": str(scene.get("pressure_if_delayed") or "The situation worsens."),
            "reveals": scene.get("reveals") if isinstance(scene.get("reveals"), list) else [],
            "conceals": scene.get("conceals") if isinstance(scene.get("conceals"), list) else [],
            "clues_available": scene.get("clues_available") if isinstance(scene.get("clues_available"), list) else [],
            "npc_present": scene.get("npc_present") if isinstance(scene.get("npc_present"), list) else [],
            "threat_level": str(scene.get("threat_level") or "tension"),
            "keeper_notes": str(scene.get("keeper_notes") or "Keep the scene concise and playable."),
        })

    return {
        "act": act_no,
        "title": payload["title"],
        "summary": payload["summary"],
        "purpose": payload["purpose"],
        "belief_shift": payload["belief_shift"],
        "required_payoffs": payload["required_payoffs"][:2],
        "module_type": module_type,
        "scenes": norm_scenes,
    }

def _descriptor_terms(doc) -> set[str]:
    meta = getattr(doc, "metadata", {}) or {}
    text = " ".join([
        str(meta.get("title_en", "") or ""),
        str(meta.get("display_name", "") or ""),
        str(meta.get("Header_2", "") or ""),
        str(meta.get("abstraction", "") or ""),
        str(meta.get("archetype", "") or ""),
        str(meta.get("role", "") or ""),
        str(meta.get("type", "") or ""),
    ]).lower()

    words = re.findall(r"[a-zA-Z0-9']+", text)
    return {
        w for w in words
        if len(w) >= 4 and w not in _DOC_TERM_STOP
    }


def _role_band(role: str) -> str:
    role = (role or "").lower()
    if any(x in role for x in ("clue", "info", "context", "investigation", "objective", "source")):
        return "investigation"
    if any(x in role for x in ("encounter", "obstacle", "danger", "threat", "pressure")):
        return "pressure"
    if any(x in role for x in ("reveal", "reversal", "escalation")):
        return "reveal"
    return "generic"

def _seed_terms(seed: str) -> list[str]:
    seed = (seed or "").lower()
    words = re.findall(r"[a-zA-Z0-9']+", seed)

    stop = {
        "the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on",
        "is", "are", "was", "were", "that", "this", "it", "they", "them",
        "into", "from", "their", "have", "has", "had", "will", "would",
        "every", "night", "exactly", "despite", "below", "constant", "operation",
        "local", "ancient", "strange", "mysterious", "broadcast", "broadcasts",
    }

    # prefer distinctive scenario nouns
    preferred = []
    for w in words:
        if len(w) < 4 or w in stop:
            continue
        preferred.append(w)

    # de-dup preserve order
    seen = set()
    out = []
    for w in preferred:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)

    return out[:10]


def _setting_terms(text: str) -> list[str]:
    raw = (text or "").lower()
    words = re.findall(r"[a-zA-Z0-9']+", raw)
    stop = {
        "present", "day", "modern", "locations", "characters", "are",
        "the", "and", "with", "from", "into", "that", "this"
    }

    seen = set()
    out = []
    for w in words:
        if len(w) < 4 or w in stop:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)

    return out[:12]


def _doc_setting_coherence(doc, era_context: str, query_text: str) -> int:
    doc_terms = _descriptor_terms(doc)
    seed_terms = set(_seed_terms(query_text))
    setting_terms = set(_setting_terms(era_context))

    score = 0

    score += 3 * len(doc_terms & setting_terms)
    score += 2 * len(doc_terms & seed_terms)

    # penalize documents that match neither the setting nor the seed
    if setting_terms and not (doc_terms & setting_terms):
        score -= 4
    if seed_terms and not (doc_terms & seed_terms):
        score -= 2

    meta = getattr(doc, "metadata", {}) or {}
    atom_type = str(meta.get("type", "") or "").lower()
    role = str(meta.get("role", "") or "").lower()

    if atom_type in ("clue", "event", "npc", "location"):
        score += 2
    if atom_type in ("appendix_character", "timeline"):
        score -= 2

    if _role_band(role) == "investigation":
        score += 2
    if _role_band(role) == "pressure":
        score += 1

    return score

def _doc_pairwise_compatibility(doc_a, doc_b) -> float:
    meta_a = getattr(doc_a, "metadata", {}) or {}
    meta_b = getattr(doc_b, "metadata", {}) or {}

    terms_a = _descriptor_terms(doc_a)
    terms_b = _descriptor_terms(doc_b)

    overlap = len(terms_a & terms_b)

    bonus = 0.0
    if str(meta_a.get("type", "")).lower() == str(meta_b.get("type", "")).lower():
        bonus += 1.0
    if str(meta_a.get("archetype", "")).lower() == str(meta_b.get("archetype", "")).lower():
        bonus += 1.5
    if _role_band(str(meta_a.get("role", ""))) == _role_band(str(meta_b.get("role", ""))):
        bonus += 1.0

    return overlap + bonus


def _doc_key(doc):
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("source", ""),
        meta.get("display_name", "") or meta.get("title_en", "") or meta.get("Header_2", ""),
        (doc.page_content or "").strip()[:120],
    )


def _select_coherent_docs(rescored, target_k: int = 4):
    if not rescored:
        return []

    pool = rescored[:10]  # keep some breadth, not whole tail
    selected = []
    seen = set()

    # 1) strongest doc first
    first_score, first_doc = pool[0]
    selected.append(first_doc)
    seen.add(_doc_key(first_doc))

    # 2) greedily add compatible docs
    while len(selected) < target_k:
        best_doc = None
        best_adjusted = None

        for base_score, doc in pool:
            key = _doc_key(doc)
            if key in seen:
                continue

            compat = sum(_doc_pairwise_compatibility(doc, s) for s in selected) / max(len(selected), 1)

            same_source_penalty = 0.0
            doc_source = (getattr(doc, "metadata", {}) or {}).get("source", "")
            if any(((getattr(s, "metadata", {}) or {}).get("source", "") == doc_source) for s in selected):
                same_source_penalty = 1.5

            adjusted = float(base_score) + compat - same_source_penalty

            if best_adjusted is None or adjusted > best_adjusted:
                best_adjusted = adjusted
                best_doc = doc

        if best_doc is None:
            break

        selected.append(best_doc)
        seen.add(_doc_key(best_doc))

    return selected


def infer_scene_stall_level(db: SessionDB) -> int:
    """
    Rough anti-loop signal: count recent user turns since the latest meaningful progress event.
    0 = fresh, 1 = mildly stalled, 2 = clearly stalled.
    """
    events = db.list_events(limit=18)
    turns_since_progress = 0

    # list_events() returns chronological order (oldest -> newest),
    # so we must scan backwards from the latest events.
    for e in reversed(events):
        et = e.get("event_type")
        if et in ("LOCATION_CHANGE", "CLUE_FOUND", "THREAD_PROGRESS"):
            break
        if et == "CHAT" and e.get("payload", {}).get("role") == "User":
            turns_since_progress += 1

    if turns_since_progress >= 4:
        return 2
    if turns_since_progress >= 2:
        return 1
    return 0


def build_stall_forcing_guard(db: SessionDB) -> str:
    stall = infer_scene_stall_level(db)
    if stall <= 0:
        return ""

    if stall == 1:
        return (
            "\n\nSTALL ESCALATION NOTICE\n"
            "The scene is starting to stall.\n"
            "This response MUST include at least one concrete payoff:\n"
            "- a clue fully revealed\n"
            "- a thread advanced\n"
            "- a location change\n"
            "- a named NPC reaction\n"
            "- a visible threat or irreversible choice\n"
            "Do not answer with another partial step toward the same discovery."
        )

    return (
        "\n\nHARD STALL OVERRIDE\n"
        "The scene is stalled.\n"
        "This response MUST NOT be another setup step, atmosphere beat, or 'you realize there may be a way'.\n"
        "You MUST do one of the following NOW:\n"
        "- reveal the core fact the investigators are pursuing\n"
        "- trigger immediate threat contact\n"
        "- force an irreversible decision\n"
        "- complete the current discovery and update the thread\n"
        "Do not suggest 'check again', 'observe further', 'try to understand more', or equivalent soft continuation."
    )


def build_scene_loop_guard(db: SessionDB) -> str:
    events = db.list_events(limit=14)
    user_actions = []
    keeper_bits = []

    for e in events:
        if e.get("event_type") != "CHAT":
            continue
        payload = e.get("payload", {})
        role = payload.get("role", "")
        content = (payload.get("content", "") or "").strip()

        if role == "User" and content:
            user_actions.append(content[:120])
        elif role == "Keeper" and content:
            keeper_bits.append(content[:220])

    recent_user = user_actions[-4:]
    recent_keeper = keeper_bits[-4:]
    keeper_blob = "\n".join(recent_keeper).lower()

    corridor_markers = [
        "door", "corridor", "hallway", "stairs", "second floor", "room",
        "двер", "корид", "сход", "кімнат", "поверх"
    ]
    corridor_hits = sum(1 for m in corridor_markers if m in keeper_blob)

    extra = (
        "\nIf the recent pattern is repeated questioning, social probing, or Psychology against the same NPC, "
        "do NOT answer with another vague read like 'they seem nervous' or 'they hide something'. "
        "Instead, the next response MUST create concrete movement through at least one of:\n"
        "- a direct contradiction or slip\n"
        "- a named clue, place, or person\n"
        "- an interruption, witness, or outside event\n"
        "- the NPC ending the exchange, leaving, threatening, or making a mistake\n"
        "- a new actionable lead that changes what the investigators do next\n"
        "A successful social/psychology payoff must yield actionable information, leverage, or a visible change in the scene."
    )

    forced = ""
    if corridor_hits >= 3:
        forced = (
            "\nHARD LOOP DETECTED: repeated room/corridor/door progression."
            "\nThe next response MUST NOT introduce another door, corridor, staircase, or adjacent room as the main development."
            "\nInstead, the next response MUST do one of the following immediately:"
            "\n- reveal the missing friend's fate or exact last known action"
            "\n- put a speaking NPC, witness, or hostile cultist on-screen"
            "\n- deliver a clue that directly points to the park and explains why"
            "\n- trigger immediate danger with a concrete consequence"
            "\n- force a decision with a cost"
        )

    return (
        "RECENT SCENE LOOP WARNING\n"
        "Do not repeat the same pattern again.\n"
        "Recent user actions:\n"
        + "\n".join(f"- {x}" for x in recent_user)
        + "\nRecent keeper outcomes:\n"
        + "\n".join(f"- {x}" for x in recent_keeper)
        + "\nThe next response MUST materially change the scene."
        + "\nForbidden next-step patterns include:"
        + "\n- another vague reaction to the same object"
        + "\n- another partial understanding of the same clue"
        + "\n- another atmospheric warning without consequence"
        + "\n- another 'you may now try X' after X was effectively already tried"
        + forced
        + "\nYou must either complete the current discovery, reveal a concrete new fact, trigger danger, or force a choice."
        + extra
    )


def build_scene_prompt(
    narrative: str,
    era: str = "",
    setting: str = "",
    visual_history: str = "",
    char_visuals: str = "",
) -> str:
    """
    Generates a Flux image prompt from the current scene narrative.
    Uses a low-temperature Gemma call for consistency.
    Imported here to keep engine.py free of nested function definitions.
    """
    from langchain_core.prompts import PromptTemplate

    tpl = PromptTemplate.from_template(
        "You are a prompt engineer for Flux, a natural-language image model. "
        "Your job is to generate a consistent visual description across a series of scenes.\n\n"
        "Setting/Era context: {era}, {setting}\n\n"
        "{char_visuals}\n\n"
        "PREVIOUS SCENE DESCRIPTIONS (for visual consistency):\n{visual_history}\n\n"
        "STEP 1 — Read the ENTIRE narrative. Identify key element or location that carries the most narrative weight.\n"
        "STEP 2 - Start with `Painterly digital illustration of...` \n"
        "STEP 3 — if narrating location - write description to recreate that environment, if narrating subject - frame it as a close or medium shot. That element must be the visual centerpiece.\n"
        "STEP 4 — Write 2-3 short natural sentences. If characters from the established list appear, "
        "describe them using their established appearance. Match era and setting in every detail.\n"
        "STEP 5 — Append exactly: 'Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.'\n\n"
        "RULES:\n"
        "- English only. No character names. No smell or sound.\n"
        "- Output ONLY the final prompt, nothing else\n\n"
        "NARRATIVE:\n{narrative}\n\nPROMPT:"
    )
    llm = get_llm(temperature=0.3)
    raw = (tpl | llm).invoke({
        "narrative": narrative,
        "era": era,
        "setting": setting,
        "visual_history": visual_history or "No previous scenes yet.",
        "char_visuals": char_visuals or "",
    }).strip()
    return " ".join(line.strip() for line in raw.splitlines() if line.strip())


async def compress_story(db: SessionDB) -> None:
    """
    Compresses the full chat history into a rolling story digest stored in
    kv_store['story_digest']. Called every N turns from handle_chat_logic.
    The digest is injected at the top of every prompt as the model's primary
    long-term memory anchor.
    """
    try:
        all_events = db.list_events(limit=60)
        chat_lines = []
        for e in all_events:
            if e.get("event_type") == "CHAT":
                p = e.get("payload", {})
                chat_lines.append(f"{p.get('role', '?').upper()}: {p.get('content', '')}")

        if len(chat_lines) < 4:
            return  # Not enough material to compress yet

        full_history = "\n\n".join(chat_lines)

        cur = db.conn.cursor()
        prev_digest = kv_get(cur, "story_digest", "(none yet)")
        lang = kv_get(cur, "language", "en")
        language_name = get_language_name(lang)

        compression_prompt = (
            f"You are a scribe summarizing a Call of Cthulhu session for continuity.\n"
            f"Language: {language_name}. Write the digest in this language.\n\n"
            f"PREVIOUS DIGEST (events before this batch):\n{prev_digest}\n\n"
            f"RECENT SESSION EXCHANGES:\n{full_history[-6000:]}\n\n"
            f"Write a STORY DIGEST — a compact, factual record of what has happened in this session.\n"
            f"Format: numbered bullet points, past tense.\n"
            f"Cover: locations visited, NPCs encountered (what was learned from/about them), clues found,\n"
            f"player decisions and their outcomes, plot threads advanced, any deaths/san loss/injuries.\n"
            f"DO NOT speculate. Only record what actually happened in the exchanges above.\n"
            f"Maximum 220 words. No preamble. Start directly with '1.'\n"
            f"DIGEST:"
        )

        llm = get_llm(temperature=0.2)
        digest = llm.invoke(compression_prompt).strip()

        cur.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('story_digest', ?)",
            (digest,)
        )
        db.conn.commit()
        logger.info(f"Story digest compressed: {len(digest)} chars")

    except Exception as e:
        logger.warning(f"Story compression failed: {e}")


async def generate_avatar_logic(req) -> dict:
    from pathlib import Path as _Path

    # _COMFY_BASE_URL = "https://provided-feeds-pipe-avatar.trycloudflare.com"
    _REQUEST_BODY_PATH = _Path(__file__).resolve().parent.parent / "img_gen" / "request_body.json"

    # Build portrait prompt
    tpl = PromptTemplate.from_template(
        "You are a prompt engineer for Flux, a natural-language image model.\n"
        "Write a portrait prompt for this character. Era/setting: {era}.\n\n"
        "Character: {occupation}, {description}\n\n"
        "RULES:\n"
        "- English only\n"
        " - Start with `Painterly digital illustration of...` \n"
        "- 1-2 sentences describing: most prominent character features matching character back-story, clothing appropriate to the era, and setting, posture/mood\n"
        "- Framing: upper body portrait, neutral or slightly dramatic background\n"
        "- No character names\n"
        "- End with exactly: 'Painterly digital illustration, pulp horror aesthetic, dramatic lighting, rich deep colors.'\n"
        "- Output ONLY the prompt, nothing else\n\n"
        "PORTRAIT PROMPT:"
    )
    llm = get_llm(temperature=0.3)
    portrait_prompt = (tpl | llm).invoke({
        "era": req.era_context or "1920s Lovecraftian Horror",
        "occupation": req.occupation,
        "description": req.physical_description
    }).strip()
    portrait_prompt = " ".join(l.strip() for l in portrait_prompt.splitlines() if l.strip())

    logger.info(f"Avatar prompt for {req.name}: {portrait_prompt}")

    try:
        from img_gen.comfy_client import ComfyClient
        _comfy = ComfyClient(_COMFY_BASE_URL)
        with open(_REQUEST_BODY_PATH, "r", encoding="utf-8") as _f:
            _body = json.load(_f)
        _body["params"]["prompt"] = portrait_prompt
        _body["params"]["width"] = 480
        _body["params"]["height"] = 640
        img_result = _comfy.generate(_body)
        return {
            "image_url": img_result["image_url"],
            "portrait_prompt": portrait_prompt
        }
    except Exception as e:
        logger.warning(f"Avatar generation failed for {req.name}: {e}")
        return {"image_url": None, "portrait_prompt": portrait_prompt}
