import { Investigator, Language, ChatResponse, AppSettings } from "../types";
import { buildInvestigator } from "./characterFactory";

const API_BASE = "/api";

export class ApiService {

  private authHeaders(): Record<string, string> {
    const token = sessionStorage.getItem('keeper_token') ?? '';
    return {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    };
  }

  public async logout(): Promise<void> {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: this.authHeaders(),
      });
    } finally {
      sessionStorage.removeItem('keeper_token');
    }
  }

  public async setLlmProvider(provider: 'ollama' | 'openai'): Promise<void> {
    await fetch(`${API_BASE}/set-provider`, {
      method: 'POST',
      headers: this.authHeaders(),
      body: JSON.stringify({ provider }),
    });
  }
  
  public async getScenarios(): Promise<{ id: string; title: string; content: string }[]> {
    try {
      const response = await fetch(`${API_BASE}/scenarios`, {
        headers: this.authHeaders(),
      });
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
        headers: this.authHeaders(),
        body: JSON.stringify({
          name,
          occupation,
          physical_description: physicalDescription,
          era_context: eraContext,
        }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.image_url || null;
    } catch {
      return null;
    }
  }

  public async generateCharacter(userPrompt: string, language: Language, eraContext?: string): Promise<Investigator> {
    const response = await fetch(`${API_BASE}/generate-character`, {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify({ prompt: userPrompt, language, era_context: eraContext }),
    });
    if (!response.ok) throw new Error("Failed to generate character");
    const charData = await response.json();
    return await buildInvestigator(charData, language, eraContext, userPrompt);
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
    const response = await fetch(`${API_BASE}/start-session`, {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify({
        investigators,
        scenarioType,
        language,
        customPrompt,
        themes,
        era_context: eraContext,
        picked_seed: pickedSeed,
      }),
    });
    if (!response.ok) throw new Error("Failed to start session");
    return await response.json();
  }

  public pollImageStatus(generationId: string): Promise<string | null> {
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const res = await fetch(`${API_BASE}/image-status/${generationId}`, {
            headers: this.authHeaders(),
          });
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
        headers: this.authHeaders(),
        body: JSON.stringify({
          session_id: "local_session",
          message,
          rag_enabled: settings.ragEnabled,
          top_k: settings.topK,
          temperature: settings.temperature,
          num_ctx: settings.numCtx,
        }),
      });
      if (!response.ok) throw new Error("Failed to send message");
      return await response.json();
    } catch (error) {
      console.error("Local API Chat Error:", error);
      return {
        narrative: "(The Keeper remains silent... connection to local Python backend failed.)",
        suggested_actions: [],
      };
    }
  }


  public streamMessage(
    message: string,
    settings: AppSettings,
    onToken: (text: string) => void,
    onDone: (result: ChatResponse) => void,
    onError: (e: unknown) => void,
  ): () => void {
    const controller = new AbortController();

    const run = async () => {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: this.authHeaders(),
        body: JSON.stringify({
          session_id: "local_session",
          message,
          rag_enabled: settings.ragEnabled,
          top_k: settings.topK,
          temperature: settings.temperature,
          num_ctx: settings.numCtx,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        onError(new Error(`HTTP ${response.status}`));
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE lines look like: "data: {...}\n\n"
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";          // keep incomplete last chunk

        for (const line of lines) {
          const text = line.replace(/^data:\s*/, "").trim();
          if (!text) continue;
          try {
            const event = JSON.parse(text);
            if (event.type === "token") {
              onToken(event.text as string);
            } else if (event.type === "done") {
              onDone(event.payload as ChatResponse);
            }
          } catch {
            // skip malformed SSE line
          }
        }
      }
    };

    run().catch((e) => {
      if ((e as any)?.name !== "AbortError") onError(e);
    });

    return () => controller.abort();
  }

  public async getScenarioBlueprint(): Promise<any> {
    const response = await fetch(`${API_BASE}/session/local_session/blueprint`, {
      headers: this.authHeaders(),
    });
    if (!response.ok) throw new Error('Blueprint not available');
    return await response.json();
  }
}

export const apiService = new ApiService();
