import { Investigator, Attributes, Skill, Language } from "../types";
import { CORE_SKILLS } from "../data/skills";

// Bilingual canonical skill pairs
const SKILL_MAP: { en: string; ru: string }[] = [
  { en: "Accounting",               ru: "Бухгалтерское дело" },
  { en: "Anthropology",             ru: "Антропология" },
  { en: "Appraise",                 ru: "Оценка" },
  { en: "Archaeology",              ru: "Археология" },
  { en: "Art/Craft",                ru: "Ис-во/ремесло" },
  { en: "Charm",                    ru: "Обаяние" },
  { en: "Climb",                    ru: "Лазание" },
  { en: "Credit Rating",            ru: "Средства" },
  { en: "Cthulhu Mythos",           ru: "Мифы Ктулху" },
  { en: "Disguise",                 ru: "Маскировка" },
  { en: "Dodge",                    ru: "Уклонение" },
  { en: "Drive Auto",               ru: "Вождение" },
  { en: "Electrical Repair",        ru: "Электрика" },
  { en: "Fast Talk",                ru: "Красноречие" },
  { en: "Fighting (Brawl)",         ru: "Ближний бой (драка)" },
  { en: "Firearms (Handgun)",       ru: "Стрельба (Пистолет)" },
  { en: "Firearms (Rifle/Shotgun)", ru: "Стрельба (Дроб./Винт.)" },
  { en: "First Aid",                ru: "Первая помощь" },
  { en: "History",                  ru: "История" },
  { en: "Intimidate",               ru: "Запугивание" },
  { en: "Jump",                     ru: "Прыжки" },
  { en: "Language (Own)",           ru: "Язык Родной" },
  { en: "Language (Other)",         ru: "Язык Иностранный" },
  { en: "Law",                      ru: "Юриспруденция" },
  { en: "Library Use",              ru: "Работа в библиотеке" },
  { en: "Listen",                   ru: "Слух" },
  { en: "Locksmith",                ru: "Взлом" },
  { en: "Mechanical Repair",        ru: "Механика" },
  { en: "Medicine",                 ru: "Медицина" },
  { en: "Natural World",            ru: "Естествознание" },
  { en: "Navigate",                 ru: "Ориентирование" },
  { en: "Occult",                   ru: "Оккультизм" },
  { en: "Persuade",                 ru: "Убеждение" },
  { en: "Pilot",                    ru: "Пилотирование" },
  { en: "Psychology",               ru: "Психология" },
  { en: "Psychoanalysis",           ru: "Психоанализ" },
  { en: "Ride",                     ru: "Верховая езда" },
  { en: "Science",                  ru: "Наука" },
  { en: "Sleight of Hand",          ru: "Ловкость рук" },
  { en: "Spot Hidden",              ru: "Внимание" },
  { en: "Stealth",                  ru: "Скрытность" },
  { en: "Survival",                 ru: "Выживание" },
  { en: "Swim",                     ru: "Плавание" },
  { en: "Throw",                    ru: "Метание" },
  { en: "Track",                    ru: "Чтение следов" },
  { en: "Computer Use",             ru: "Работа с компьютером" },
];

/**
 * Given any skill name (EN or RU), returns the canonical name in target language.
 * Returns null if no match — means it's a custom/extra skill.
 */
function normalizeSkillName(name: string, language: Language): string | null {
  const lower = name.toLowerCase().trim();
  const match = SKILL_MAP.find(
    pair => pair.en.toLowerCase() === lower || pair.ru.toLowerCase() === lower
  );
  if (!match) return null;
  return language === 'ru' ? match.ru : match.en;
}

/**
 * Build the full investigator from raw LLM character data.
 * - Starts with ALL CORE_SKILLS at base values (same as official sheet)
 * - Overlays LLM-trained skills on top (taking higher value)
 * - Normalizes all skill names to target language
 * - Appends custom/extra skills the LLM invented that aren't in core list
 */
export function buildInvestigator(charData: any, language: Language): Investigator {
  const chars = charData.characteristics;

  // --- Attributes ---
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

  // --- Skills ---
  // Step 1: start from the full canonical list in target language (base values)
  const baseList = language === 'ru' ? CORE_SKILLS.ru : CORE_SKILLS.en;
  const skillMap = new Map<string, number>(
    baseList.map(s => {
      // Resolve dynamic base values
      if (s.name === 'Уклонение' || s.name === 'Dodge')
        return [s.name, Math.floor(chars.DEX / 2)];
      if (s.name === 'Язык Родной' || s.name === 'Language (Own)')
        return [s.name, chars.EDU];
      return [s.name, s.value];
    })
  );

  // Step 2: overlay LLM skills — normalize name to target language, take max value
  const customSkills: Skill[] = [];

  for (const s of (charData.skills || [])) {
    const canonical = normalizeSkillName(s.name, language);

    if (canonical) {
      // Known skill: apply dynamic overrides, then take max(base, llm)
      let value = s.value;
      if (canonical === 'Уклонение' || canonical === 'Dodge')
        value = Math.floor(chars.DEX / 2);
      if (canonical === 'Язык Родной' || canonical === 'Language (Own)')
        value = chars.EDU;

      const current = skillMap.get(canonical) ?? 0;
      skillMap.set(canonical, Math.max(current, value));
    } else {
      // Custom skill (e.g. "Oceanography", "Neural Interface") — keep in target language as-is
      customSkills.push({ name: s.name, value: s.value });
    }
  }

  // Step 3: convert map back to sorted array, append custom skills at end
  const finalSkills: Skill[] = Array.from(skillMap.entries())
    .map(([name, value]) => ({ name, value }));

  // Custom skills go after the core list
  finalSkills.push(...customSkills);

  return { ...charData, attributes, skills: finalSkills };
}