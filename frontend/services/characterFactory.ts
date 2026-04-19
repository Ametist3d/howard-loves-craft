import { Investigator, Attributes, Skill, Language } from "../types";
import { CORE_SKILLS } from "../data/skills";
import { pipeline } from "@huggingface/transformers";

type CanonicalSkill = string;

type SkillPlan = {
  occupation_archetype?: string;
  priority_skills?: string[];
  personal_interest_skills?: string[];
  credit_rating_target?: number;
};

type ArchetypeConfig = {
  occupationSkills: CanonicalSkill[];
  defaultInterestSkills: CanonicalSkill[];
  creditRatingRange: [number, number];
  defaultCreditTarget: number;
};

const CANONICAL_SKILL_NAMES = new Set<string>(CORE_SKILLS.en.map(s => s.name));

const ARCHETYPE_CONFIG: Record<string, ArchetypeConfig> = {
  scholar: {
    occupationSkills: [
      "Library Use",
      "History",
      "Language (Other)",
      "Psychology",
      "Anthropology",
      "Occult",
      "Spot Hidden",
      "Credit Rating",
    ],
    defaultInterestSkills: ["First Aid", "Listen", "Persuade", "Natural World"],
    creditRatingRange: [10, 40],
    defaultCreditTarget: 20,
  },

  field_operator: {
    occupationSkills: [
      "Survival",
      "Spot Hidden",
      "Listen",
      "Navigate",
      "Track",
      "First Aid",
      "Stealth",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Natural World", "Climb", "Swim", "Throw"],
    creditRatingRange: [5, 25],
    defaultCreditTarget: 12,
  },

  field_scholar: {
    occupationSkills: [
      "Navigate",
      "Spot Hidden",
      "Natural World",
      "Survival",
      "Library Use",
      "History",
      "Language (Other)",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Occult", "Psychology", "First Aid", "Mechanical Repair"],
    creditRatingRange: [8, 35],
    defaultCreditTarget: 18,
  },

  technical_specialist: {
    occupationSkills: [
      "Computer Use",
      "Electrical Repair",
      "Mechanical Repair",
      "Science",
      "Spot Hidden",
      "Library Use",
      "Navigate",
      "Credit Rating",
    ],
    defaultInterestSkills: ["First Aid", "Psychology", "Listen", "Locksmith"],
    creditRatingRange: [10, 40],
    defaultCreditTarget: 18,
  },

  social_operator: {
    occupationSkills: [
      "Persuade",
      "Fast Talk",
      "Charm",
      "Psychology",
      "Intimidate",
      "Spot Hidden",
      "Law",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Listen", "Stealth", "Library Use", "Occult"],
    creditRatingRange: [10, 60],
    defaultCreditTarget: 25,
  },

  covert_operator: {
    occupationSkills: [
      "Stealth",
      "Sleight of Hand",
      "Locksmith",
      "Spot Hidden",
      "Fast Talk",
      "Drive Auto",
      "Listen",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Psychology", "First Aid", "Fighting (Brawl)", "Navigate"],
    creditRatingRange: [5, 30],
    defaultCreditTarget: 10,
  },

  authority_custodian: {
    occupationSkills: [
      "Law",
      "Psychology",
      "Intimidate",
      "Listen",
      "Spot Hidden",
      "Persuade",
      "History",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Library Use", "First Aid", "Firearms (Handgun)", "Drive Auto"],
    creditRatingRange: [15, 55],
    defaultCreditTarget: 28,
  },

  healer: {
    occupationSkills: [
      "Medicine",
      "First Aid",
      "Psychology",
      "Science",
      "Library Use",
      "Spot Hidden",
      "Listen",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Occult", "Language (Other)", "Natural World", "Persuade"],
    creditRatingRange: [15, 50],
    defaultCreditTarget: 25,
  },

  esoteric_contact: {
    occupationSkills: [
      "Occult",
      "Psychology",
      "Listen",
      "Language (Other)",
      "History",
      "Spot Hidden",
      "Persuade",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Library Use", "First Aid", "Natural World", "Stealth"],
    creditRatingRange: [5, 35],
    defaultCreditTarget: 14,
  },

  observer: {
    occupationSkills: [
      "Spot Hidden",
      "Listen",
      "Psychology",
      "Library Use",
      "Language (Other)",
      "History",
      "Stealth",
      "Credit Rating",
    ],
    defaultInterestSkills: ["Occult", "First Aid", "Natural World", "Persuade"],
    creditRatingRange: [8, 35],
    defaultCreditTarget: 16,
  },
};

function normalizeSkillNameEn(name: string): string | null {
  const lower = name.toLowerCase().trim();

  const aliases: Record<string, string> = {
    "accounting": "Accounting",
    "anthropology": "Anthropology",
    "appraise": "Appraise",
    "archaeology": "Archaeology",
    "art/craft": "Art/Craft",
    "charm": "Charm",
    "climb": "Climb",
    "credit rating": "Credit Rating",
    "cthulhu mythos": "Cthulhu Mythos",
    "disguise": "Disguise",
    "dodge": "Dodge",
    "drive auto": "Drive Auto",
    "electrical repair": "Electrical Repair",
    "fast talk": "Fast Talk",
    "fighting (brawl)": "Fighting (Brawl)",
    "firearms (handgun)": "Firearms (Handgun)",
    "firearms (rifle/shotgun)": "Firearms (Rifle/Shotgun)",
    "first aid": "First Aid",
    "history": "History",
    "intimidate": "Intimidate",
    "jump": "Jump",
    "language (own)": "Language (Own)",
    "language (other)": "Language (Other)",
    "law": "Law",
    "library use": "Library Use",
    "listen": "Listen",
    "locksmith": "Locksmith",
    "mechanical repair": "Mechanical Repair",
    "medicine": "Medicine",
    "natural world": "Natural World",
    "navigate": "Navigate",
    "occult": "Occult",
    "persuade": "Persuade",
    "pilot": "Pilot",
    "psychology": "Psychology",
    "psychoanalysis": "Psychoanalysis",
    "ride": "Ride",
    "science": "Science",
    "sleight of hand": "Sleight of Hand",
    "spot hidden": "Spot Hidden",
    "stealth": "Stealth",
    "survival": "Survival",
    "swim": "Swim",
    "throw": "Throw",
    "track": "Track",
    "computer use": "Computer Use",
  };

  return aliases[lower] ?? null;
}

// type SkillFloorMap = Record<string, number>;

type EmbeddingExtractor = any;

let extractorPromise: Promise<EmbeddingExtractor> | null = null;
let conceptEmbeddingsPromise: Promise<Map<string, number[][]>> | null = null;

const CONCEPT_PROFILES = {
  field_operator: {
    prototypes: [
      "survivor, scout, guard, courier, warden, drifter, pathfinder, practical field operator under danger",
      "someone used to exposure, travel, pursuit, vigilance, and rough conditions",
      "person who notices danger early and keeps moving in hostile ground"
    ],
    boosts: {
      "Survival": 12,
      "Spot Hidden": 10,
      "Listen": 8,
      "Track": 8,
      "Navigate": 8,
      "First Aid": 6,
      "Stealth": 6,
      "Fighting (Brawl)": 6
    }
  },

  technical_analyst: {
    prototypes: [
      "codebreaker, data analyst, engineer, systems expert, technical specialist, diagnostics and signal analysis",
      "person who works with computers, devices, radio, infrastructure, technical failures, and system logic",
      "specialist in decoding, electronics, machinery, and practical troubleshooting"
    ],
    boosts: {
      "Computer Use": 14,
      "Electrical Repair": 10,
      "Mechanical Repair": 8,
      "Science": 8,
      "Library Use": 4,
      "Spot Hidden": 4
    }
  },

  scholar_investigator: {
    prototypes: [
      "researcher, archivist, chronicler, interpreter, skeptical scholar, investigator of records and meaning",
      "person who studies documents, languages, archives, testimony, and historical or cultural clues",
      "analytical mind working through comparison, context, and evidence"
    ],
    boosts: {
      "Library Use": 14,
      "History": 10,
      "Psychology": 6,
      "Language (Other)": 6,
      "Anthropology": 6,
      "Occult": 4
    }
  },

  social_operator: {
    prototypes: [
      "fixer, negotiator, recruiter, confessor, provocateur, social manipulator, dealmaker",
      "person who influences, pressures, persuades, recruits, calms, or extracts information from others",
      "operator who works through people, leverage, trust, lies, and negotiations"
    ],
    boosts: {
      "Persuade": 12,
      "Fast Talk": 10,
      "Charm": 8,
      "Psychology": 8,
      "Intimidate": 6,
      "Credit Rating": 4,
      "Law": 4
    }
  },

  covert_operator: {
    prototypes: [
      "smuggler, undocumented runner, courier, debt collector, illicit fixer, quiet criminal operator",
      "person used to hidden movement, smuggling, lock access, lying, and staying unseen",
      "operator in gray zones, evasion, concealment, and practical street survival"
    ],
    boosts: {
      "Stealth": 10,
      "Sleight of Hand": 8,
      "Locksmith": 8,
      "Fast Talk": 6,
      "Drive Auto": 6,
      "Spot Hidden": 4,
      "Fighting (Brawl)": 4
    }
  },

  expedition_specialist: {
    prototypes: [
      "cartographer, relic hunter, pathfinder, field researcher, remote explorer, expedition specialist",
      "person working in ruins, wilderness, routes, mapping, terrain, and physical evidence in the field",
      "explorer combining navigation, survival, material clues, and old sites"
    ],
    boosts: {
      "Navigate": 12,
      "Archaeology": 10,
      "Natural World": 8,
      "Survival": 8,
      "Spot Hidden": 8,
      "Climb": 4,
      "Track": 4
    }
  },

  esoteric_contact: {
    prototypes: [
      "cult defector, apostate, dream interpreter, doubtful initiate, person close to forbidden belief systems",
      "someone with exposure to confessions, visions, symbols, rites, dreams, or hidden spiritual structures",
      "person who recognizes strange patterns before ordinary people do"
    ],
    boosts: {
      "Occult": 12,
      "Psychology": 8,
      "Listen": 4,
      "Language (Other)": 4,
      "History": 4
    }
  }
};

async function getExtractor(): Promise<EmbeddingExtractor> {
  if (!extractorPromise) {
    extractorPromise = pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );
  }
  return extractorPromise;
}

async function embedTexts(texts: string[]): Promise<number[][]> {
  const extractor = await getExtractor();
  const output = await extractor(texts, {
    pooling: "mean",
    normalize: true,
  });
  return output.tolist() as number[][];
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

async function getConceptEmbeddings(): Promise<Map<string, number[][]>> {
  if (!conceptEmbeddingsPromise) {
    conceptEmbeddingsPromise = (async () => {
      const map = new Map<string, number[][]>();
      for (const [concept, profile] of Object.entries(CONCEPT_PROFILES)) {
        map.set(concept, await embedTexts(profile.prototypes));
      }
      return map;
    })();
  }
  return conceptEmbeddingsPromise;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function scoreConcept(inputEmbedding: number[], prototypeEmbeddings: number[][]): number {
  const sims = prototypeEmbeddings.map(v => dot(inputEmbedding, v));
  return Math.max(...sims);
}

async function applySemanticSkillBoosts(
  skillMap: Map<string, number>,
  charData: any,
  eraContext?: string,
  userPrompt?: string
): Promise<void> {
  const textBlob = [
    userPrompt ?? "",
    eraContext ?? "",
    charData?.occupation ?? "",
    charData?.backstory ?? "",
    charData?.description ?? "",
    charData?.physical_description ?? "",
  ].filter(Boolean).join("\n");

  if (!textBlob.trim()) return;

  const [inputEmbedding] = await embedTexts([textBlob]);
  const conceptEmbeddings = await getConceptEmbeddings();

  for (const [concept, profile] of Object.entries(CONCEPT_PROFILES)) {
    const prototypes = conceptEmbeddings.get(concept);
    if (!prototypes) continue;

    const rawScore = scoreConcept(inputEmbedding, prototypes);

    // ignore weak matches; smoothly scale stronger ones
    const weight = clamp01((rawScore - 0.35) / 0.35);
    if (weight <= 0) continue;

    for (const [skillName, boost] of Object.entries(profile.boosts)) {
      const current = skillMap.get(skillName) ?? 0;
      const delta = Math.round(boost * weight);
      if (delta <= 0) continue;
      skillMap.set(skillName, Math.min(85, current + delta));
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(value)));
}

function uniq<T>(items: T[]): T[] {
  return Array.from(new Set(items));
}

function roll3d6x5(): number {
  const d = () => Math.floor(Math.random() * 6) + 1;
  return (d() + d() + d()) * 5;
}

function normalizeSkillList(skills: unknown): CanonicalSkill[] {
  if (!Array.isArray(skills)) return [];
  return uniq(
    skills
      .map(s => normalizeSkillNameEn(String(s)))
      .filter((s): s is string => !!s && CANONICAL_SKILL_NAMES.has(s))
  );
}

function getArchetypeConfig(archetypeRaw?: string): ArchetypeConfig {
  const key = String(archetypeRaw || "").trim().toLowerCase();
  return ARCHETYPE_CONFIG[key] ?? ARCHETYPE_CONFIG.scholar;
}

function calcMoveRate(chars: any, age: number): number {
  let mov = 8;

  if (chars.STR < chars.SIZ && chars.DEX < chars.SIZ) mov = 7;
  else if (chars.STR > chars.SIZ && chars.DEX > chars.SIZ) mov = 9;
  else mov = 8;

  if (age >= 40 && age < 50) mov -= 1;
  else if (age >= 50 && age < 60) mov -= 2;
  else if (age >= 60 && age < 70) mov -= 3;
  else if (age >= 70 && age < 80) mov -= 4;
  else if (age >= 80) mov -= 5;

  return Math.max(1, mov);
}

function calcBuildAndDamageBonus(str: number, siz: number): { build: number; damageBonus: string } {
  const total = str + siz;

  if (total <= 64) return { build: -2, damageBonus: "-2" };
  if (total <= 84) return { build: -1, damageBonus: "-1" };
  if (total <= 124) return { build: 0, damageBonus: "+0" };
  if (total <= 164) return { build: 1, damageBonus: "+1D4" };
  if (total <= 204) return { build: 2, damageBonus: "+1D6" };

  const over = total - 205;
  const steps = Math.floor(over / 80);
  const build = 3 + steps;
  const dice = 2 + steps;
  return { build, damageBonus: `+${dice}D6` };
}

function buildAttributes(chars: any, age: number): Attributes {
  const hpMax = Math.floor((chars.CON + chars.SIZ) / 10);
  const sanityStart = chars.POW;
  const mpMax = Math.floor(chars.POW / 5);
  const luck = roll3d6x5();
  const { build, damageBonus } = calcBuildAndDamageBonus(chars.STR, chars.SIZ);

  return {
    HP: { current: hpMax, max: hpMax },
    Sanity: { current: sanityStart, max: 99, start: sanityStart },
    MagicPoints: { current: mpMax, max: mpMax },
    Luck: { current: luck, max: 99 },
    MoveRate: calcMoveRate(chars, age),
    Build: build,
    DamageBonus: damageBonus,
  };
}

function buildBaseSkillMap(chars: any): Map<string, number> {
  return new Map<string, number>(
    CORE_SKILLS.en.map(s => {
      if (s.name === "Dodge") return [s.name, Math.floor(chars.DEX / 2)];
      if (s.name === "Language (Own)") return [s.name, chars.EDU];
      return [s.name, s.value];
    })
  );
}

function getOccupationBudget(archetypeRaw: string | undefined, chars: any): number {
  const archetype = String(archetypeRaw || "").trim().toLowerCase();

  switch (archetype) {
    case "scholar":
      return chars.EDU * 4;
    case "field_operator":
      return chars.EDU * 2 + chars.DEX * 2;
    case "field_scholar":
      return chars.EDU * 2 + chars.INT * 2;
    case "technical_specialist":
      return chars.EDU * 2 + chars.INT * 2;
    case "social_operator":
      return chars.EDU * 2 + chars.APP * 2;
    case "covert_operator":
      return chars.EDU * 2 + chars.DEX * 2;
    case "authority_custodian":
      return chars.EDU * 2 + chars.POW * 2;
    case "healer":
      return chars.EDU * 2 + chars.INT * 2;
    case "esoteric_contact":
      return chars.EDU * 2 + chars.POW * 2;
    case "observer":
      return chars.EDU * 2 + chars.INT * 2;
    default:
      return chars.EDU * 4;
  }
}

function mergeOccupationSkills(planSkills: CanonicalSkill[], cfg: ArchetypeConfig): CanonicalSkill[] {
  return uniq(
    [...planSkills, ...cfg.occupationSkills]
      .filter(s => s !== "Cthulhu Mythos")
  );
}

function mergeInterestSkills(planSkills: CanonicalSkill[], cfg: ArchetypeConfig): CanonicalSkill[] {
  return uniq(
    [...planSkills, ...cfg.defaultInterestSkills]
      .filter(s => s !== "Cthulhu Mythos" && s !== "Credit Rating")
  );
}

function allocateWeightedPoints(
  skillMap: Map<string, number>,
  skills: CanonicalSkill[],
  budget: number,
  caps: Partial<Record<CanonicalSkill, number>> = {},
  weights?: number[]
): void {
  const validSkills = skills.filter(s => CANONICAL_SKILL_NAMES.has(s));
  if (!validSkills.length || budget <= 0) return;

  const localWeights =
    weights && weights.length === validSkills.length
      ? weights
      : validSkills.map((_, idx) => Math.max(0.5, 1.8 - idx * 0.18));

  const weightSum = localWeights.reduce((a, b) => a + b, 0);
  if (weightSum <= 0) return;

  let spent = 0;

  validSkills.forEach((skill, idx) => {
    const current = skillMap.get(skill) ?? 0;
    const cap = caps[skill] ?? 75;
    if (current >= cap) return;

    const rawShare = Math.round((budget * localWeights[idx]) / weightSum);
    const add = Math.min(rawShare, Math.max(0, cap - current));
    if (add > 0) {
      skillMap.set(skill, current + add);
      spent += add;
    }
  });

  let remainder = budget - spent;
  let guard = 0;

  while (remainder > 0 && guard < 500) {
    guard += 1;
    let progressed = false;

    for (const skill of validSkills) {
      if (remainder <= 0) break;
      const current = skillMap.get(skill) ?? 0;
      const cap = caps[skill] ?? 75;
      if (current >= cap) continue;

      skillMap.set(skill, current + 1);
      remainder -= 1;
      progressed = true;
    }

    if (!progressed) break;
  }
}

function applyCreditRating(
  skillMap: Map<string, number>,
  budget: number,
  requestedTarget: number | undefined,
  range: [number, number],
  fallbackTarget: number
): number {
  const target = clamp(
    Number.isFinite(requestedTarget) ? Number(requestedTarget) : fallbackTarget,
    range[0],
    range[1]
  );

  const current = skillMap.get("Credit Rating") ?? 0;
  const needed = Math.max(0, target - current);
  const used = Math.min(needed, Math.max(0, budget));

  skillMap.set("Credit Rating", current + used);
  return used;
}

function applyLlmSkillHints(skillMap: Map<string, number>, llmSkills: any[]): void {
  for (const s of (llmSkills || [])) {
    const canonical = normalizeSkillNameEn(String(s?.name || ""));
    if (!canonical || canonical === "Credit Rating" || canonical === "Cthulhu Mythos") continue;
    if (!CANONICAL_SKILL_NAMES.has(canonical)) continue;

    let hint = Number(s?.value || 0);
    if (!Number.isFinite(hint) || hint <= 0) continue;

    const current = skillMap.get(canonical) ?? 0;
    if (hint <= current) continue;

    const softTarget = Math.min(hint, current + 8);
    skillMap.set(canonical, softTarget);
  }
}

function finalizeDerivedSkills(skillMap: Map<string, number>, chars: any): void {
  skillMap.set("Dodge", Math.floor(chars.DEX / 2));
  skillMap.set("Language (Own)", chars.EDU);
}

function ensureMinimumUsefulSkillFloor(
  skillMap: Map<string, number>,
  archetypeRaw: string | undefined
): void {
  const archetype = String(archetypeRaw || "").trim().toLowerCase();

  const floors: Record<string, number> = {
    "Spot Hidden": 25,
  };

  if (archetype === "field_operator" || archetype === "field_scholar") {
    floors["Navigate"] = 35;
    floors["Survival"] = 35;
    floors["Natural World"] = 25;
  }

  if (archetype === "technical_specialist") {
    floors["Computer Use"] = 35;
    floors["Electrical Repair"] = 25;
    floors["Mechanical Repair"] = 20;
    floors["Science"] = 25;
  }

  if (archetype === "healer") {
    floors["Medicine"] = 40;
    floors["First Aid"] = 45;
  }

  if (archetype === "social_operator") {
    floors["Persuade"] = 40;
    floors["Psychology"] = 35;
  }

  if (archetype === "scholar" || archetype === "field_scholar") {
    floors["Library Use"] = 40;
    floors["History"] = 25;
  }

  for (const [skill, floor] of Object.entries(floors)) {
    const current = skillMap.get(skill) ?? 0;
    if (current < floor) skillMap.set(skill, floor);
  }
}

function getSkillPlan(charData: any): SkillPlan {
  const plan = charData?.skill_plan || {};
  return {
    occupation_archetype: plan.occupation_archetype,
    priority_skills: Array.isArray(plan.priority_skills) ? plan.priority_skills : [],
    personal_interest_skills: Array.isArray(plan.personal_interest_skills) ? plan.personal_interest_skills : [],
    credit_rating_target: typeof plan.credit_rating_target === "number" ? plan.credit_rating_target : undefined,
  };
}

async function applySkillPlanAllocation(
  skillMap: Map<string, number>,
  charData: any,
  _eraContext?: string,
  _userPrompt?: string
): Promise<void> {
  const chars = charData?.characteristics || {};
  const plan = getSkillPlan(charData);

  const cfg = getArchetypeConfig(plan.occupation_archetype);

  const occupationSkills = mergeOccupationSkills(
    normalizeSkillList(plan.priority_skills),
    cfg
  );

  const interestSkills = mergeInterestSkills(
    normalizeSkillList(plan.personal_interest_skills),
    cfg
  );

  const occupationBudget = getOccupationBudget(plan.occupation_archetype, chars);
  const personalInterestBudget = Math.max(0, Number(chars.INT || 0) * 2);

  let remainingOccupationBudget = occupationBudget;

  // Credit Rating is part of occupation points
  const creditSpent = applyCreditRating(
    skillMap,
    remainingOccupationBudget,
    plan.credit_rating_target,
    cfg.creditRatingRange,
    cfg.defaultCreditTarget
  );
  remainingOccupationBudget -= creditSpent;

  allocateWeightedPoints(
    skillMap,
    occupationSkills.filter(s => s !== "Credit Rating"),
    remainingOccupationBudget,
    {
      "Library Use": 75,
      "History": 70,
      "Language (Other)": 70,
      "Psychology": 70,
      "Anthropology": 65,
      "Occult": 55,
      "Spot Hidden": 70,
      "Navigate": 75,
      "Natural World": 70,
      "Survival": 70,
      "Computer Use": 75,
      "Electrical Repair": 70,
      "Mechanical Repair": 70,
      "Science": 75,
      "Persuade": 70,
      "Fast Talk": 70,
      "Charm": 65,
      "Intimidate": 65,
      "Law": 60,
      "Stealth": 70,
      "Sleight of Hand": 65,
      "Locksmith": 65,
      "Drive Auto": 60,
      "Medicine": 75,
      "First Aid": 75,
      "Listen": 65,
      "Track": 65,
    }
  );

  allocateWeightedPoints(
    skillMap,
    interestSkills,
    personalInterestBudget,
    {
      "Occult": 55,
      "Psychology": 65,
      "First Aid": 65,
      "Mechanical Repair": 55,
      "Natural World": 60,
      "Listen": 60,
      "Spot Hidden": 70,
      "Stealth": 60,
      "Drive Auto": 55,
      "Swim": 55,
      "Throw": 55,
      "Track": 60,
      "Library Use": 65,
      "History": 60,
      "Navigate": 60,
      "Computer Use": 60,
      "Science": 60,
      "Medicine": 65,
      "Persuade": 60,
      "Fast Talk": 60,
      "Charm": 60,
    }
  );

  ensureMinimumUsefulSkillFloor(skillMap, plan.occupation_archetype);
  applyLlmSkillHints(skillMap, charData.skills || []);
}


export async function buildInvestigator(
  charData: any,
  _language: Language,
  eraContext?: string,
  userPrompt?: string
): Promise<Investigator> {
  const chars = charData.characteristics;
  const age = Number(charData.age || 30);

  const attributes: Attributes = buildAttributes(chars, age);
  const skillMap = buildBaseSkillMap(chars);

  for (const s of (charData.skills || [])) {
    const canonical = normalizeSkillNameEn(s.name);
    if (!canonical) continue;

    let value = s.value;
    if (canonical === "Dodge") value = Math.floor(chars.DEX / 2);
    if (canonical === "Language (Own)") value = chars.EDU;

    const current = skillMap.get(canonical) ?? 0;
    skillMap.set(canonical, Math.max(current, value));
  }

  // INSERT NEW LOGIC HERE
  await applySkillPlanAllocation(skillMap, charData, eraContext, userPrompt);
  await applySemanticSkillBoosts(skillMap, charData, eraContext, userPrompt);

  // Re-lock derived skills after boosts
  skillMap.set("Dodge", Math.floor(chars.DEX / 2));
  skillMap.set("Language (Own)", chars.EDU);

  const finalSkills: Skill[] = Array.from(skillMap.entries())
    .map(([name, value]) => ({ name, value: clamp(value, 0, 99) }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return {
    ...charData,
    attributes,
    skills: finalSkills,
  };
}

