export type Language = 'en' | 'ru';

export interface PrebuiltScenario {
  id: string;
  title: string;
  content: string;
}

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
  status?: 'ok' | 'injured' | 'dead' | 'insane';
}

export type InvestigatorConfig = Pick<Investigator, 'id' | 'name' | 'occupation' | 'background'> & { id: number };

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
  imageGenerating?: boolean;
}


export interface AppSettings {
  ragEnabled: boolean;
  topK: number;
  temperature: number;
  numCtx: number;
}
export interface GameState {
  phase: 'setup' | 'loading' | 'playing';
  investigators: Investigator[];
  scenarioTitle: string;
  language: Language;
  prebuiltScenario?: PrebuiltScenario | null;
  llmProvider: 'ollama' | 'openai';
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
  image_url?: string;
  generation_id?: string;
  updated_actor?: {
    name: string;
    hp: number;
    san: number;
    mp: number;
    status: string;
  };
}