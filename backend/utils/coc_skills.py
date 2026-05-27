# backend/utils/coc_skills.py

from __future__ import annotations

from typing import Dict


CORE_SKILL_BASES: Dict[str, int] = {
    "Accounting": 5,
    "Anthropology": 1,
    "Appraise": 5,
    "Archaeology": 1,
    "Art/Craft": 5,
    "Charm": 15,
    "Climb": 20,
    "Credit Rating": 0,
    "Cthulhu Mythos": 0,
    "Disguise": 5,
    "Dodge": 0,  # dynamic: DEX / 2
    "Drive Auto": 20,
    "Electrical Repair": 10,
    "Fast Talk": 5,
    "Fighting (Brawl)": 25,
    "Firearms (Handgun)": 20,
    "Firearms (Rifle/Shotgun)": 25,
    "First Aid": 30,
    "History": 5,
    "Intimidate": 15,
    "Jump": 20,
    "Language (Own)": 0,  # dynamic: EDU
    "Language (Other)": 1,
    "Law": 5,
    "Library Use": 20,
    "Listen": 20,
    "Locksmith": 1,
    "Mechanical Repair": 10,
    "Medicine": 1,
    "Natural World": 10,
    "Navigate": 10,
    "Occult": 5,
    "Persuade": 10,
    "Pilot": 1,
    "Psychology": 10,
    "Psychoanalysis": 1,
    "Ride": 5,
    "Science": 1,
    "Sleight of Hand": 10,
    "Spot Hidden": 25,
    "Stealth": 20,
    "Survival": 10,
    "Swim": 20,
    "Throw": 20,
    "Track": 10,

    # You already use this in backend + characterFactory.
    # Keep it for modern scenarios.
    "Computer Use": 5,

    # Backend already references this in TECHNICAL_ROLL_TARGETS.
    "Operate Heavy Machinery": 1,
}


CHARACTERISTIC_TARGETS = ("STR", "CON", "SIZ", "DEX", "APP", "INT", "POW", "EDU")


SKILL_ALIASES = {
    "own language": "Language (Own)",
    "language (own)": "Language (Own)",
    "native language": "Language (Own)",
    "computer": "Computer Use",
    "computer use": "Computer Use",
    "firearms": "Firearms (Handgun)",
    "handgun": "Firearms (Handgun)",
    "pistol": "Firearms (Handgun)",
    "rifle": "Firearms (Rifle/Shotgun)",
    "shotgun": "Firearms (Rifle/Shotgun)",
    "cthulhu mythos": "Cthulhu Mythos",
}

SKILL_GLOSS: dict[str, str] = {
    "STR": "raw strength force lift break push pull hold wrestle overpower",
    "CON": "endurance resist poison disease fatigue pain cold heat suffocation",
    "SIZ": "body size mass bulk reach weight physical scale",
    "DEX": "agility reflex balance coordination quick hands precision movement",
    "APP": "appearance presence attractiveness first impression social bearing",
    "INT": "idea logic deduction reasoning insight connect facts understand plan",
    "POW": "willpower mental resistance sanity mental pressure psychic influence",
    "EDU": "formal knowledge education general scholarship academic recall",

    "Listen": "listen hear overhear eavesdrop faint sound muffled voices conversation noise",
    "Spot Hidden": "notice spot search visual clue hidden detail trace anomaly see observe",
    "Track": "track follow footprints trail traces movement signs path",
    "Navigate": "navigate route direction map orientation wayfinding bearings",

    "Psychology": "read motives emotion fear lie deception behavior mental state intent",
    "Persuade": "convince reason negotiate cooperate appeal argument persuade",
    "Fast Talk": "bluff improvise lie mislead quick excuse talk past pressure verbally",
    "Charm": "charm rapport warmth friendly trust flirt social ease",
    "Intimidate": "threaten demand scare coerce pressure menace force compliance",

    "Library Use": "research archives records documents books indexes newspaper files",
    "History": "historical context past events age origin tradition chronology",
    "Occult": "occult ritual myth esoteric supernatural cult symbol magic folklore",
    "Archaeology": "ancient artifact ruin pottery ceramic excavation burial relic inscription",
    "Anthropology": "culture customs tribe society ritual human groups ethnography",
    "Language (Other)": "translate foreign language decipher text inscription runes unknown writing",
    "Language (Own)": "read native language grammar writing composition exact wording",
    "Art/Craft": "art craft technique style painting sculpture music forgery handmade object",
    "Appraise": "value price authenticity quality material worth identify valuable object",
    "Science": "scientific analysis measurement experiment theory anomaly physics biology chemistry",

    "Mechanical Repair": "repair machine mechanism gears pipes engine device mechanical fault",
    "Electrical Repair": "repair wiring circuit battery radio transmitter power electrical fault",
    "Computer Use": "computer terminal digital logs software access data system",
    "Locksmith": "lock pick open locked door safe latch keys disable lock",

    "First Aid": "emergency treatment stabilize wound bleeding bandage immediate injury",
    "Medicine": "diagnose illness symptoms pathology surgery poison disease treatment",
    "Natural World": "plants animals weather ecology nature terrain survival environment",

    "Stealth": "sneak hide move quietly unseen avoid notice shadow silent",
    "Climb": "climb ascend descend wall rope cliff difficult surface",
    "Jump": "jump leap gap vault obstacle distance landing",
    "Swim": "swim water current dive stay afloat drowning",
    "Dodge": "dodge evade avoid attack hazard reflex sidestep",
    "Throw": "throw toss hurl aim object projectile",
    "Fighting (Brawl)": "melee punch kick grapple wrestle close combat disarm tackle",
    "Firearms": "shoot firearm gun pistol rifle shotgun aim fire",
    "Firearms (Handgun)": "shoot handgun pistol revolver aim fire",
    "Firearms (Rifle/Shotgun)": "shoot rifle shotgun long gun aim fire",

    "Drive Auto": "drive car truck vehicle road steering",
    "Pilot": "pilot aircraft boat vehicle controls steering navigation",
    "Ride": "ride horse animal mount control saddle",
    "Operate Heavy Machinery": "operate crane excavator heavy machinery industrial controls",
    "Cthulhu Mythos": "cthulhu mythos forbidden cosmic truth elder gods unnatural entities sanity-shattering revelation",

    "Disguise": "disguise appearance costume impersonate look like someone else",
    "Sleight of Hand": "palming pickpocket conceal small object manual trick",
    "Credit Rating": "wealth status social class funds financial access reputation",
    "Accounting": "books ledger accounts finances audit money trail",
    "Law": "legal rules court police warrant contract regulation",
    "Survival": "survive wilderness shelter food fire exposure hostile environment",
    "Psychoanalysis": "therapy calm madness long mental treatment trauma",
}

def normalize_roll_target_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""

    lowered = raw.lower().strip()
    if lowered in SKILL_ALIASES:
        return SKILL_ALIASES[lowered]

    for skill in CORE_SKILL_BASES:
        if lowered == skill.lower():
            return skill

    for stat in CHARACTERISTIC_TARGETS:
        if lowered == stat.lower():
            return stat

    return raw


def all_core_roll_targets() -> list[str]:
    return list(CORE_SKILL_BASES.keys()) + list(CHARACTERISTIC_TARGETS)


def base_skill_value(skill_name: str, *, stats: dict | None = None) -> int:
    skill = normalize_roll_target_name(skill_name)
    stats = stats or {}

    if skill == "Dodge":
        return max(0, int(stats.get("DEX") or stats.get("dex") or 0) // 2)

    if skill == "Language (Own)":
        return max(0, int(stats.get("EDU") or stats.get("edu") or 0))

    return int(CORE_SKILL_BASES.get(skill, 0))
