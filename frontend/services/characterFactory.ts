import { Investigator, Attributes, Skill, Language } from "../types";
import { CORE_SKILLS } from "../data/skills";

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

export function buildInvestigator(charData: any, _language: Language): Investigator {
  const chars = charData.characteristics;

  const hpMax = Math.floor((chars.CON + chars.SIZ) / 10);
  const sanityStart = chars.POW;
  const mpMax = Math.floor(chars.POW / 5);
  const luck = (Math.floor(Math.random() * 6) + 1 +
                Math.floor(Math.random() * 6) + 1 +
                Math.floor(Math.random() * 6) + 1) * 5;

  const attributes: Attributes = {
    HP:          { current: hpMax,        max: hpMax },
    Sanity:      { current: sanityStart,  max: 99, start: sanityStart },
    MagicPoints: { current: mpMax,        max: mpMax },
    Luck:        { current: luck,         max: 99 },
    MoveRate: 8,
    Build: 0,
    DamageBonus: "+0"
  };

  // Always use canonical English core list
  const skillMap = new Map<string, number>(
    CORE_SKILLS.en.map(s => {
      if (s.name === "Dodge") return [s.name, Math.floor(chars.DEX / 2)];
      if (s.name === "Language (Own)") return [s.name, chars.EDU];
      return [s.name, s.value];
    })
  );

  // Overlay only recognized English canonical skills
  for (const s of (charData.skills || [])) {
    const canonical = normalizeSkillNameEn(s.name);
    if (!canonical) continue;

    let value = s.value;
    if (canonical === "Dodge") value = Math.floor(chars.DEX / 2);
    if (canonical === "Language (Own)") value = chars.EDU;

    const current = skillMap.get(canonical) ?? 0;
    skillMap.set(canonical, Math.max(current, value));
  }

  const finalSkills: Skill[] = Array.from(skillMap.entries())
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return {
    ...charData,
    attributes,
    skills: finalSkills,
  };
}