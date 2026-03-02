export type Language = 'en' | 'ru';

export interface Attribute {
  current: number;
  max: number;
  start?: number;
}

export interface Characteristics {
  STR: number;
  CON: number;
  SIZ: number;
  DEX: number;
  APP: number;
  INT: number;
  POW: number;
  EDU: number;
}

export interface Attributes {
  HP: Attribute;
  Sanity: Attribute;
  MagicPoints: Attribute;
  Luck: Attribute;
  MoveRate: number;
  Build: number;
  DamageBonus: string;
}

export interface Skill {
  name: string;
  value: number;
}

export interface Investigator {
  id?: number;
  name: string;
  occupation: string;
  age: number;
  residence: string;
  birthplace: string;
  characteristics: Characteristics;
  attributes: Attributes;
  skills: Skill[];
  inventory: string[];
  background: string;
  physical_description: string;
  avatarUrl?: string;
}

export enum MessageSender {
  SYSTEM = 'system',
  USER = 'user',
  KEEPER = 'keeper'
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  content: string;
  timestamp: number;
  image?: string;
}


export interface AppSettings {
  ragEnabled: boolean;
  topK: number;
  temperature: number;
}
export interface GameState {
  phase: 'setup' | 'loading' | 'playing';
  investigators: Investigator[];
  scenarioTitle: string;
  language: Language;
  settings: AppSettings;
}

export interface ChatResponse {
  narrative: string;
  visual_prompt?: string;
  suggested_actions: string[];
  state_updates?: {
    character_name?: string;
    hp_change?: number;
    sanity_change?: number;
    mp_change?: number;
    luck_change?: number;
    inventory_add?: string;
    inventory_remove?: string;
  };
}

