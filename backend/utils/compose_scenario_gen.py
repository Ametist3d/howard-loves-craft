from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence
#pylint: disable=import-error

# These defaults mirror the current act module library.
EARLY_MODULES = [
    "arrival_and_briefing",
    "surface_inquiry",
    "witness_network",
    "site_investigation",
    "institutional_obstruction",
    "parallel_leads",
    "false_pattern_strengthens",
    "danger_probe",
    "hidden_logistics",
    "social_crack",
]

MID_MODULES = [
    "parallel_leads",
    "false_pattern_strengthens",
    "danger_probe",
    "hidden_logistics",
    "social_crack",
    "contradiction_reveal",
    "betrayal_or_complicity",
    "reversal_of_cause",
    "negotiated_window",
    "countdown_crisis",
    "chase_or_containment",
    "converging_threads",
]

FINAL_MODULES = [
    "costly_climax",
    "countdown_crisis",
    "chase_or_containment",
    "negotiated_window",
    "aftermath_choice",
]

MODULE_PROFILE = {
    "arrival_and_briefing": {
        "purpose": "Establish the surface situation, immediate stakes, authority posture, and the first actionable lead.",
        "summary": "An opening act built around arrival, briefing, or first contact with the ordinary problem before the deeper truth is visible.",
        "belief_shift": "Investigators move from surface orientation to suspicion that the obvious situation is incomplete.",
        "required_payoffs": ["first actionable lead", "first ordinary explanation"],
        "scene_functions": ["hook", "social pressure", "clue reveal", "misdirection"],
    },
    "surface_inquiry": {
        "purpose": "Test obvious witnesses, locations, and explanations before the deeper structure becomes clear.",
        "summary": "Investigators compare the most accessible accounts and evidence, building an initial theory that still may be wrong.",
        "belief_shift": "What first seemed simple now acquires friction, inconsistency, or hidden motive.",
        "required_payoffs": ["surface contradiction", "one viable wrong theory"],
        "scene_functions": ["investigation", "misdirection", "social conflict", "clue reveal"],
    },
    "witness_network": {
        "purpose": "Build the case through incompatible testimonies, rumor chains, and social texture rather than pure physical evidence.",
        "summary": "Different witnesses describe the same disturbance in ways that cannot all be true, forcing comparison and interpretation.",
        "belief_shift": "The truth appears distributed across people who each see only part of the pattern.",
        "required_payoffs": ["conflicting testimony", "social leverage or distrust"],
        "scene_functions": ["social conflict", "investigation", "misdirection", "choice"],
    },
    "site_investigation": {
        "purpose": "Advance the scenario through place-based evidence, mechanism, architecture, and environmental anomaly.",
        "summary": "Physical sites reveal repeated use, practical operations, or impossible details that social stories cannot fully explain.",
        "belief_shift": "The site itself becomes a witness, showing structure behind the rumor.",
        "required_payoffs": ["material clue", "evidence that changes the theory"],
        "scene_functions": ["investigation", "clue reveal", "danger escalation", "obstruction"],
    },
    "institutional_obstruction": {
        "purpose": "Create friction through official secrecy, procedure, ownership, law, or reputation management.",
        "summary": "Human systems now resist the investigators, making access itself part of the mystery.",
        "belief_shift": "The problem is not only hidden by the uncanny, but also by ordinary power structures.",
        "required_payoffs": ["blocked access", "human motive for concealment"],
        "scene_functions": ["social conflict", "obstruction", "choice", "danger escalation"],
    },
    "parallel_leads": {
        "purpose": "Create at least two meaningful leads that can be pursued in different order and recontextualize each other.",
        "summary": "The case branches into parallel lines of inquiry that each become clearer when compared against the other.",
        "belief_shift": "Investigators stop following one breadcrumb chain and begin mapping a larger system.",
        "required_payoffs": ["two viable threads", "cross-thread contradiction"],
        "scene_functions": ["investigation", "choice", "clue reveal", "social conflict"],
    },
    "false_pattern_strengthens": {
        "purpose": "Let the wrong explanation become more convincing before overturning it later.",
        "summary": "Evidence appears to confirm the misleading theory, raising the cost of abandoning it.",
        "belief_shift": "Investigators commit to an explanation that later evidence will crack open.",
        "required_payoffs": ["false lead strengthened", "future contradiction seeded"],
        "scene_functions": ["misdirection", "investigation", "social pressure", "danger escalation"],
    },
    "danger_probe": {
        "purpose": "Trade passive investigation for risky contact, entry, descent, trespass, or exposure.",
        "summary": "The truth advances because investigators push into a dangerous space rather than because another witness calmly explains it.",
        "belief_shift": "The case is no longer safe to study from the outside.",
        "required_payoffs": ["risk under pressure", "evidence gained at a price"],
        "scene_functions": ["danger escalation", "combat", "investigation", "clue reveal"],
    },
    "hidden_logistics": {
        "purpose": "Reveal the repeated practical operations behind the disturbance: routes, timing, feeding, maintenance, supply, storage, signaling, or labor.",
        "summary": "The mystery is shown to be a working system, not a one-off supernatural outburst.",
        "belief_shift": "Investigators stop seeing isolated weirdness and start seeing organized process.",
        "required_payoffs": ["operational pattern", "evidence of repeated use"],
        "scene_functions": ["investigation", "clue reveal", "contradiction", "danger escalation"],
    },
    "social_crack": {
        "purpose": "An NPC, witness, or authority buckles, slips, confesses, or reveals personal pressure.",
        "summary": "The social mask fractures and a human contradiction becomes visible.",
        "belief_shift": "The investigators learn that fear, compromise, or guilt is shaping the case from inside.",
        "required_payoffs": ["NPC crack", "motive under pressure"],
        "scene_functions": ["social conflict", "reversal", "betrayal", "clue reveal"],
    },
    "contradiction_reveal": {
        "purpose": "Materially overturn an earlier assumption with direct contradiction.",
        "summary": "Something previously treated as stable or explanatory can no longer stand after the new evidence.",
        "belief_shift": "The old theory breaks, forcing a new mental model.",
        "required_payoffs": ["major contradiction", "old theory broken"],
        "scene_functions": ["contradiction", "reversal", "investigation", "danger escalation"],
    },
    "betrayal_or_complicity": {
        "purpose": "Reveal that an important person enabled, rationalized, concealed, or misread the danger.",
        "summary": "A key human relationship is reinterpreted through complicity, betrayal, or protective secrecy.",
        "belief_shift": "The investigators see that human choice helped sustain the hidden structure.",
        "required_payoffs": ["human complicity", "trust rupture"],
        "scene_functions": ["betrayal", "social conflict", "reversal", "choice"],
    },
    "reversal_of_cause": {
        "purpose": "Reveal that the assumed cause is actually an effect, response, trigger, or cover symptom.",
        "summary": "The meaning of the central anomaly flips: what looked primary is secondary, and what seemed secondary is foundational.",
        "belief_shift": "The investigators now understand the system differently enough that later action must change.",
        "required_payoffs": ["cause/effect reversal", "new operative truth"],
        "scene_functions": ["reversal", "reassessment", "choice", "contradiction"],
    },
    "negotiated_window": {
        "purpose": "Create a temporary chance for contact, terms, delay, exchange, or partial understanding.",
        "summary": "The scenario opens a narrow chance to influence events without immediate all-out destruction.",
        "belief_shift": "Not every path forward is pure attack or flight; negotiation becomes possible but costly.",
        "required_payoffs": ["contact opportunity", "terms or implied bargain"],
        "scene_functions": ["choice", "social conflict", "reassessment", "danger escalation"],
    },
    "countdown_crisis": {
        "purpose": "Make delay itself unacceptable through a deadline that threatens life, exposure, or irreversible loss.",
        "summary": "The case enters a countdown state in which hesitation is now a decision with cost.",
        "belief_shift": "Investigators move from learning to urgent resolution.",
        "required_payoffs": ["deadline active", "delay cost visible"],
        "scene_functions": ["hard choice", "danger escalation", "climax", "final confrontation"],
    },
    "chase_or_containment": {
        "purpose": "Force pursuit, interception, isolation, rerouting, or containment instead of another clue chain.",
        "summary": "The scenario becomes procedural and urgent: catch it, block it, redirect it, or lose control of it.",
        "belief_shift": "Understanding must now become action.",
        "required_payoffs": ["active pursuit or containment", "immediate consequence"],
        "scene_functions": ["danger escalation", "combat", "choice", "climax"],
    },
    "converging_threads": {
        "purpose": "Bring multiple leads into one revealed hidden structure.",
        "summary": "Independent threads now point at the same underlying engine, actor, or process.",
        "belief_shift": "Fragmented evidence resolves into one coherent architecture of truth.",
        "required_payoffs": ["threads converge", "shared hidden structure"],
        "scene_functions": ["reassessment", "contradiction", "choice", "clue reveal"],
    },
    "costly_climax": {
        "purpose": "Deliver a real endgame choice with sacrifice, compromise, collateral damage, secrecy, exposure, or moral burden.",
        "summary": "All threads converge into a final decision where no clean resolution is free.",
        "belief_shift": "The investigators know enough to act, but not enough to avoid paying for it.",
        "required_payoffs": ["multiple viable outcomes", "explicit cost or tradeoff"],
        "scene_functions": ["climax", "hard choice", "final confrontation", "danger escalation"],
    },
    "aftermath_choice": {
        "purpose": "Handle concealment, fallout, moral burden, reprisals, future threat, or cover story after the main crisis breaks.",
        "summary": "The immediate danger may be past, but the scenario still demands a choice about what truth, guilt, or future consequence remains.",
        "belief_shift": "Resolution becomes burden rather than pure victory.",
        "required_payoffs": ["fallout handled", "future consequence framed"],
        "scene_functions": ["choice", "reassessment", "social pressure", "aftermath"],
    },
}

SCENE_TEMPLATE = '''        {{{{
          "scene": "Scene {scene_no} name",
          "location": "Choose from the reusable location graph; locations may repeat across scenes and acts",
          "scene_function": "{scene_function}",
          "dramatic_question": "What uncertainty, pressure, or decision defines this scene",
          "entry_condition": "Why this scene becomes available now",
          "exit_condition": "What must be learned, decided, or suffered before moving on",
          "trigger": "What starts this scene or makes it unavoidable",
          "description": "Atmospheric arrival description grounded in the current setting (2-4 sentences)",
          "what_happens": "Key development for this scene, shaped by the act module's function (2-4 sentences)",
          "pressure_if_delayed": "What gets worse if investigators hesitate here",
          "reveals": ["What this scene can reveal"],
          "conceals": ["What still remains hidden here"],
          "clues_available": ["Clue title"],
          "npc_present": ["NPC name"],
          "threat_level": "none|tension|danger|combat",
          "keeper_notes": "Behind-the-scenes guidance for emphasis, concealment, redirection, or consequence"
        }}}}'''


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _pick_act_modules(rng: random.Random, act_count: int) -> list[str]:
    modules: list[str] = []
    used = set()

    first = rng.choice(EARLY_MODULES)
    modules.append(first)
    used.add(first)

    while len(modules) < act_count - 1:
        choices = [m for m in MID_MODULES if m not in used] or MID_MODULES[:]
        chosen = rng.choice(choices)
        modules.append(chosen)
        used.add(chosen)

    final_choices = [m for m in FINAL_MODULES if m not in used] or FINAL_MODULES[:]
    modules.append(rng.choice(final_choices))

    if "false_pattern_strengthens" not in modules and act_count >= 3:
        modules[min(1, act_count - 2)] = "false_pattern_strengthens"
    if not any(m in modules for m in ("contradiction_reveal", "betrayal_or_complicity", "reversal_of_cause", "converging_threads")):
        modules[max(1, act_count - 2)] = rng.choice([
            "contradiction_reveal", "betrayal_or_complicity", "reversal_of_cause", "converging_threads"
        ])
    if modules[-1] not in FINAL_MODULES:
        modules[-1] = rng.choice(FINAL_MODULES)

    return modules


def _scene_counts(rng: random.Random, act_count: int) -> list[int]:
    counts = [rng.randint(1, 4) for _ in range(act_count)]
    total = sum(counts)
    while total < max(4, act_count + 2):
        idx = rng.randrange(act_count)
        if counts[idx] < 4:
            counts[idx] += 1
            total += 1
    while total > 12:
        idx = rng.randrange(act_count)
        if counts[idx] > 1:
            counts[idx] -= 1
            total -= 1
    return counts


def _act_object(act_no: int, module: str, scene_count: int, rng: random.Random) -> str:
    profile = MODULE_PROFILE[module]
    scene_functions: Sequence[str] = profile["scene_functions"]
    scenes = [
        SCENE_TEMPLATE.format(scene_no=i + 1, scene_function=rng.choice(scene_functions))
        for i in range(scene_count)
    ]
    scenes_text = ",\n".join(scenes)
    payoffs = ",\n        ".join(f'"{x}"' for x in profile["required_payoffs"])
    return f'''    {{{{
      "act": {act_no},
      "title": "Act {act_no} title",
      "summary": "{profile['summary']}",
      "purpose": "{profile['purpose']}",
      "belief_shift": "{profile['belief_shift']}",
      "required_payoffs": [
        {payoffs}
      ],
      "scenes": [
{scenes_text}
      ]
    }}}}'''


def build_dynamic_json_contract(*, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    act_count = rng.randint(2, 5)
    modules = _pick_act_modules(rng, act_count)
    scene_counts = _scene_counts(rng, act_count)
    total_scenes = sum(scene_counts)
    acts_text = ",\n".join(
        _act_object(i + 1, modules[i], scene_counts[i], rng) for i in range(act_count)
    )

    planning_rules = f'''STRUCTURE RULES
- Generate {act_count} acts.
- Scene distribution by act: {scene_counts}.
- Total scene count must be {total_scenes}.
- Use the act skeleton shown below, but fill each act's summary, purpose, belief_shift, and required_payoffs according to its chosen act module type.
- The chosen act module sequence for this run is: {", ".join(modules)}.
- Locations form a connected reusable graph, not a one-location-per-scene corridor.
- The same named location may appear in multiple scenes and across multiple acts.
- Scenes may reopen old places with new pressure, altered access, new NPC presence, or changed meaning.
- Keep the same JSON field names shown below. Do not rename, remove, or replace them.
- Acts 1..N should escalate from surface situation toward contradiction, recontextualization, pressure, and costly resolution.
- The final act must still deliver more than one viable resolution, cost, or tradeoff.
- Do not add extra top-level sections beyond the schema below.
'''

    schema = f'''OUTPUT STRICT JSON ONLY (no markdown, no text outside braces):
{{{{
  "title": "Short evocative scenario title",
  "era_and_setting": "Specific time and place (1-2 sentences)",
  "atmosphere_notes": "Dominant sensory/emotional tone for the Keeper (1 sentence)",
  "inciting_hook": "What draws investigators in — the surface-level call to action (1-2 sentences)",
  "core_mystery": "The central hidden structure investigators uncover (1-2 sentences)",
  "hidden_threat": "The Mythos entity, force, system, or agency at work, its goal, and its method (2-3 sentences)",
  "truth_the_players_never_suspect": "The single biggest recontextualizing truth (1-2 sentences)",
  "scenario_engine": {{{{
    "surface_explanation": "The most plausible early explanation that seems true at first",
    "actual_explanation": "What is really happening underneath the surface situation",
    "false_leads": [
      "False lead 1",
      "False lead 2"
    ],
    "contradictions": [
      "Contradiction 1",
      "Contradiction 2"
    ],
    "reversals": [
      "Reversal 1",
      "Reversal 2"
    ],
    "dynamic_pressures": [
      "time pressure",
      "social or institutional pressure"
    ],
    "climax_choices": [
      {{{{
        "option": "A possible late-stage solution or compromise",
        "cost": "What it costs immediately",
        "consequence": "What happens if chosen"
      }}}},
      {{{{
        "option": "Another viable solution or compromise",
        "cost": "What it costs",
        "consequence": "What happens if chosen"
      }}}}
    ]
  }}}},
  "acts": [
{acts_text}
  ],
  "locations": [
    {{{{
      "name": "Reusable location name",
      "tags": "entry|danger|clue|lair|social|transit|authority",
      "hidden": "The single most important concealed fact about this place"
    }}}}
  ],
  "npcs": [
    {{{{
      "name": "Name",
      "role": "ally|neutral|enemy|hidden_enemy",
      "secret": "What they truly are or hide",
      "motivation": "What they want and why"
    }}}}
  ],
  "clues": [
    {{{{
      "title": "Clue name",
      "content": "Surface meaning of the clue",
      "true_meaning": "What it actually implies",
      "location": "Reusable location name"
    }}}}
  ],
  "plot_threads": [
    {{{{
      "name": "Thread name",
      "stakes": "What happens if investigators fail or ignore this thread",
      "steps": 3
    }}}}
  ]
}}}}'''

    return planning_rules + "\n" + schema


def build_final_prompt(
    *,
    instructions_path: Path,
    hook_path: Path,
    act_path: Path,
    resolution_path: Path,
    rng: random.Random | None = None,
) -> str:
    instructions = _read(instructions_path)
    hook_types = _read(hook_path)
    act_modules = _read(act_path)
    resolution_types = _read(resolution_path)
    contract = build_dynamic_json_contract(rng=rng)

    parts = [
        instructions,
        "",
        "HOOK TYPE LIBRARY",
        hook_types,
        "",
        "ACT MODULE TYPE LIBRARY",
        act_modules,
        "",
        "RESOLUTION TYPE LIBRARY",
        resolution_types,
        "",
        contract,
        "",
    ]
    return "\n".join(parts)


def compose_and_write_scenario_prompt(*, prompts_root: Path, rng: random.Random | None = None) -> Path:
    scenario_dir = prompts_root / "scenario"
    out_path = prompts_root / "scenario_gen.txt"

    final_prompt = build_final_prompt(
        instructions_path=scenario_dir / "scenario_instructions.txt",
        hook_path=scenario_dir / "hook_type.txt",
        act_path=scenario_dir / "act_module_type.txt",
        resolution_path=scenario_dir / "resolution_type.txt",
        rng=rng,
    )
    out_path.write_text(final_prompt, encoding="utf-8")
    return out_path


def prepare_scenario_prompt_for_session(*, prompts_root: Path, session_id: str, language: str) -> str:
    from utils.prompt_translate import ensure_translated_prompts

    compose_and_write_scenario_prompt(prompts_root=prompts_root)

    translated_root = prompts_root.parent / "data" / "sessions" / "translated_prompts"
    translated_scenario_gen = translated_root / session_id / language / "scenario_gen.txt"
    if translated_scenario_gen.exists():
        translated_scenario_gen.unlink()

    return ensure_translated_prompts(
        session_id,
        language,
        refresh_filenames=("scenario_gen.txt",),
    )
