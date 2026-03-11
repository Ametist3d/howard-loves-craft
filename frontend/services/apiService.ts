import { Investigator, Attributes, Skill, Language, ChatResponse, AppSettings } from "../types";
import { buildInvestigator } from "./characterFactory";

const API_BASE = "/api";

export class ApiService {

  public async getScenarios(): Promise<{ id: string; title: string; content: string }[]> {
    try {
      const response = await fetch(`${API_BASE}/scenarios`);
      if (!response.ok) return [];
      return await response.json();
    } catch {
      return [];
    }
  }

  public async generateAvatar(
    name: string,
    occupation: string,
    physicalDescription: string,
    eraContext?: string
  ): Promise<string | null> {
    try {
      const res = await fetch(`${API_BASE}/generate-avatar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          occupation,
          physical_description: physicalDescription,
          era_context: eraContext
        })
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.image_url || null;
    } catch {
      return null;
    }
  }

  public async generateCharacter(userPrompt: string, language: Language, eraContext?: string): Promise<Investigator> {
    try {
      const response = await fetch(`${API_BASE}/generate-character`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userPrompt, language, era_context: eraContext })
      });

      if (!response.ok) throw new Error("Failed to generate character");
      const charData = await response.json();

      return buildInvestigator(charData, language);
    } catch (error) {
      console.error("Local API Gen Character Error:", error);
      throw error;
    }
  }

  public async startSession(
    investigators: Investigator[],
    scenarioType: string,
    language: Language,
    customPrompt?: string,
    themes?: string[],
    eraContext?: string,
    pickedSeed?: string
  ): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE}/start-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          investigators,
          scenarioType,
          language,
          customPrompt,
          themes,
          era_context: eraContext,
          picked_seed: pickedSeed
        })
      });

      if (!response.ok) throw new Error("Failed to start session");
      return await response.json();
    } catch (error) {
      console.error("Local API Start Session Error:", error);
      throw error;
    }
  }

  pollImageStatus(generationId: string): Promise<string | null> {
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const res = await fetch(`${API_BASE}/image-status/${generationId}`);
          if (!res.ok) { resolve(null); return; }
          const data = await res.json();
          if (data.ready) { resolve(data.image_url); return; }
        } catch {
          resolve(null); return;
        }
        setTimeout(poll, 2000); 
      };
      setTimeout(poll, 2000);
    });
  }

  public async sendMessage(message: string, settings: AppSettings): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "local_session",
          message,
          rag_enabled: settings.ragEnabled,
          top_k: settings.topK,
          temperature: settings.temperature,
          num_ctx: settings.numCtx
        })
      });

      if (!response.ok) throw new Error("Failed to send message");
      return await response.json();
    } catch (error) {
      console.error("Local API Chat Error:", error);
      return {
        narrative: "(The Keeper remains silent... connection to local Python backend failed.)",
        suggested_actions: []
      };
    }
  }

  public async getScenarioBlueprint(): Promise<any> {
    const response = await fetch(`${API_BASE}/session/local_session/blueprint`);
    if (!response.ok) throw new Error('Blueprint not available');
    return await response.json();
  }
}

export const apiService = new ApiService();