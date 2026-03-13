import React, { useState, useCallback } from 'react';
import { SetupScreen } from './components/SetupScreen';
import { ChatInterface } from './components/ChatInterface';
import { CharacterSheet } from './components/CharacterSheet';
import { DiceRoller } from './components/DiceRoller';
import { SettingsModal } from './components/SettingsModal';
import { AppSettings } from './types';
import { GameState, ChatMessage, MessageSender, Language, ChatResponse, PrebuiltScenario, InvestigatorConfig } from './types';
import { apiService } from './services/apiService';
import { SCENARIO_SEEDS } from './data/scenarios';
import { AuthScreen } from './components/AuthScreen';

const INITIAL_STATE: GameState = {
  phase: 'setup',
  investigators: [],
  scenarioTitle: '',
  language: 'ru',
  llmProvider: 'ollama',
  settings: {
    ragEnabled: true,
    topK: 2,
    temperature: 0.7,
    numCtx: 16384 
  }
};

function App() {
    const [authToken, setAuthToken] = useState<string | null>(
      sessionStorage.getItem('keeper_token')
    );
    const [gameState, setGameState] = useState<GameState>(INITIAL_STATE);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showSheet, setShowSheet] = useState(false);
    const [suggestedActions, setSuggestedActions] = useState<string[]>([]);
    const [showSettings, setShowSettings] = useState(false);

  const addMessage = (sender: MessageSender, content: string, id?: string) => {
    const newMsg = {
      id: id || Date.now().toString() + Math.random(),
      sender,
      content,
      timestamp: Date.now()
    };
    setMessages((prev) => [...prev, newMsg]);
    return newMsg.id;
  };

  const handleStateUpdate = useCallback((response: any) => {
    // Accept either a raw state_updates object or a full response with updated_actor.
    // updated_actor carries ABSOLUTE values from the backend (idempotent — safe to call twice).
    // Falls back to delta-apply only for luck (never tracked server-side).
    const updates = response?.state_updates ?? response;
    const absolute = response?.updated_actor;

    if (!updates && !absolute) return;
    const targetName = (absolute?.name ?? updates?.character_name ?? "").toLowerCase();
    if (!targetName) return;

    // Collect display labels BEFORE entering the state updater (no side-effects inside updater)
    const labels: string[] = [];
    if (absolute) {
      // We know the absolute new values — build labels from state_updates deltas for display
      if (updates?.hp_change)     labels.push(`HP ${updates.hp_change > 0 ? '+' : ''}${updates.hp_change}`);
      if (updates?.sanity_change) labels.push(`SAN ${updates.sanity_change > 0 ? '+' : ''}${updates.sanity_change}`);
      if (updates?.mp_change)     labels.push(`MP ${updates.mp_change > 0 ? '+' : ''}${updates.mp_change}`);
    }

    setGameState((prev) => {
      if (prev.investigators.length === 0) return prev;
      const index = prev.investigators.findIndex(inv =>
        inv.name.toLowerCase().includes(targetName) || targetName.includes(inv.name.toLowerCase())
      );
      if (index === -1) return prev;

      const updatedInvestigators = [...prev.investigators];
      const inv = { ...updatedInvestigators[index] };
      const attr = { ...inv.attributes };

      if (absolute) {
        // ✅ SET absolute values from backend — idempotent, no double-apply possible
        if (absolute.hp  !== undefined) attr.HP.current          = Math.min(attr.HP.max,         Math.max(0, absolute.hp));
        if (absolute.san !== undefined) attr.Sanity.current      = Math.min(attr.Sanity.max,      Math.max(0, absolute.san));
        if (absolute.mp  !== undefined) attr.MagicPoints.current = Math.min(attr.MagicPoints.max, Math.max(0, absolute.mp));
      } else if (updates) {
        // Fallback delta-apply (only reached when backend sends no updated_actor, e.g. luck)
        if (updates.luck_change) attr.Luck.current = Math.min(attr.Luck.max, Math.max(0, attr.Luck.current + updates.luck_change));
      }

      let inventory = [...inv.inventory];
      if (updates?.inventory_add) inventory.push(updates.inventory_add);
      if (updates?.inventory_remove) {
        const idx = inventory.findIndex(i => i.toLowerCase().includes(updates.inventory_remove.toLowerCase()));
        if (idx > -1) inventory.splice(idx, 1);
      }

      updatedInvestigators[index] = { ...inv, attributes: attr, inventory,
        status: absolute?.status ?? inv.status };
      return { ...prev, investigators: updatedInvestigators };
    });

    // Side-effects OUTSIDE the state updater (React StrictMode safe)
    if (labels.length > 0) {
      addMessage(MessageSender.SYSTEM, `[${targetName}: ${labels.join(', ')}]`);
    }
    if (absolute?.status === 'dead') {
      addMessage(MessageSender.SYSTEM,
        `💀 ${absolute.name} погиб. Их история на этом заканчивается.`);
    } else if (absolute?.status === 'insane') {
      addMessage(MessageSender.SYSTEM,
        `🌀 ${absolute.name} сошёл с ума. Рассудок покинул их навсегда.`);
    }
  }, []);

  const processAiResponse = async (response: ChatResponse) => {
    const msgId = Date.now().toString() + Math.random();

    // 1. Show narrative immediately
    setMessages((prev) => [...prev, {
      id: msgId,
      sender: MessageSender.KEEPER,
      content: response.narrative,
      timestamp: Date.now(),
      image: undefined,
      imageGenerating: !!response.generation_id,  // ← drives the spinner
    }]);

    setSuggestedActions(response.suggested_actions || []);
    if (response.state_updates || response.updated_actor) handleStateUpdate(response);

    // 2. Poll for image — fully detached, input is unblocked immediately
    if (response.generation_id) {
      apiService.pollImageStatus(response.generation_id).then(imageUrl => {
        setMessages((prev) => prev.map((m) =>
          m.id === msgId
            ? { ...m, image: imageUrl || undefined, imageGenerating: false }
            : m
        ));
      });
    }
  };

  const handleStartGame = async (config: {
    investigators: InvestigatorConfig[];
    scenario: 'prebuilt' | 'random' | 'custom';
    customPrompt: string;
    themes?: string[];
    language: Language;
    prebuiltScenario?: PrebuiltScenario | null;
    llmProvider: 'ollama' | 'openai';
  }) => {
    setIsLoading(true);
    try {
      // 1. Resolve era context and picked seed based on scenario type
      let eraContext = '1920s Lovecraftian Horror';
      let pickedSeed = '';
      let scenarioTitle = 'Unknown';

      if (config.scenario === 'prebuilt' && config.prebuiltScenario) {
        // Use the full scenario text as the seed for RAG — sharp and specific
        pickedSeed  = config.prebuiltScenario.content;
        scenarioTitle = config.prebuiltScenario.title;
        eraContext  = 'Derive era and setting strictly from the prebuilt scenario text. Do not invent or substitute a different era.';
      } else if (config.scenario === 'random' && config.themes && config.themes.length > 0) {
        eraContext = config.themes
          .map(t => SCENARIO_SEEDS[t]?.[0]?.replace('ERA:', '').trim() ?? '')
          .filter(Boolean)
          .join(' | ');
        const pool = config.themes.flatMap(t => SCENARIO_SEEDS[t]?.slice(1) ?? []);
        pickedSeed = pool[Math.floor(Math.random() * pool.length)] ?? '';
        scenarioTitle = `Random: ${config.themes.join(', ')}`;
      } else if (config.scenario === 'custom' && config.customPrompt) {
        pickedSeed    = config.customPrompt;
        eraContext    = 'Derive era and setting from the custom prompt provided.';
        scenarioTitle = 'Custom Scenario';
      }

      // 2. Generate investigators (pass era so character gen matches setting)
      const promises = config.investigators.map(data => {
        const prompt = `Name: ${data.name}, Job: ${data.occupation}. Background: ${data.background}`;
        return apiService.generateCharacter(prompt, config.language, eraContext);
      });
      const generatedInvestigators = await Promise.all(promises);

      setGameState(prev => ({
        ...prev,
        phase: 'loading',
        investigators: generatedInvestigators,
        scenarioTitle,
        language: config.language
      }));
      await apiService.setLlmProvider(config.llmProvider);
      // 3. Start session — pass resolved era + seed to backend
      const result = await apiService.startSession(
        generatedInvestigators,
        config.scenario,
        config.language,
        config.customPrompt,
        config.themes,
        eraContext,
        pickedSeed
      );

      setGameState(prev => ({ ...prev, phase: 'playing' }));
      await processAiResponse(result);
      
      // Generate avatars in background — fires after game starts, patches in as ready
      generatedInvestigators.forEach(async (inv) => {
        if (!inv.physical_description) return;
        const avatarUrl = await apiService.generateAvatar(
          inv.name,
          inv.occupation,
          inv.physical_description,
          eraContext
        );
        if (avatarUrl) {
          setGameState(prev => ({
            ...prev,
            investigators: prev.investigators.map(i =>
              i.name === inv.name ? { ...i, avatarUrl } : i
            )
          }));
        }
      });

    } catch (error) {
      console.error("Init error:", error);
      addMessage(MessageSender.SYSTEM, "Error initializing game. Is the Python backend running?");
      setGameState(INITIAL_STATE);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async (e?: React.FormEvent, overrideText?: string) => {
    e?.preventDefault();
    const textToSend = overrideText || input;
    if (!textToSend.trim() || isLoading) return;

    setInput('');
    setSuggestedActions([]);
    addMessage(MessageSender.USER, textToSend);
    setIsLoading(true);

    if (textToSend.startsWith('/')) {
      if (textToSend === '/inventory') setShowSheet(true);
      setIsLoading(false);
      return;
    }

    const allDead = gameState.investigators.length > 0 &&
      gameState.investigators.every(inv =>
        inv.status === 'dead' || inv.attributes.HP.current === 0
      );
    if (allDead) {
      addMessage(MessageSender.SYSTEM,
        '💀 Все следователи мертвы. Игра окончена.');
      setIsLoading(false);
      return;
    }

    // Create a placeholder Keeper message that we'll fill in token by token
    const msgId = Date.now().toString() + Math.random();
    setMessages(prev => [...prev, {
      id: msgId,
      sender: MessageSender.KEEPER,
      content: '',
      timestamp: Date.now(),
      imageGenerating: false,
    }]);

    apiService.streamMessage(
      textToSend,
      gameState.settings,
      // onToken — append each chunk to the message
      (token: string) => {
        setMessages(prev => prev.map(m =>
          m.id === msgId ? { ...m, content: m.content + token } : m
        ));
      },
      // onDone — swap raw streamed text for clean parsed narrative, apply state
      async (result) => {
        setIsLoading(false);
        // Replace streamed raw JSON with the clean parsed narrative
        setMessages(prev => prev.map(m =>
          m.id === msgId
            ? { ...m, content: result.narrative ?? m.content, imageGenerating: !!result.generation_id }
            : m
        ));
        setSuggestedActions(result.suggested_actions || []);
        if (result.state_updates || result.updated_actor) handleStateUpdate(result);
        // Poll for image — detached, does not block input
        if (result.generation_id) {
          apiService.pollImageStatus(result.generation_id).then(imageUrl => {
            setMessages(prev => prev.map(m =>
              m.id === msgId ? { ...m, image: imageUrl || undefined, imageGenerating: false } : m
            ));
          });
        }
      },
      // onError
      (error) => {
        console.error("Stream error:", error);
        setMessages(prev => prev.map(m =>
          m.id === msgId ? { ...m, content: "(The Keeper's voice fades... connection lost.)" } : m
        ));
        setIsLoading(false);
      }
    );
  };

  const handleUseLuck = useCallback((investigatorName: string, luckSpent: number) => {
      setGameState(prev => ({
          ...prev,
          investigators: prev.investigators.map(inv => {
              if (inv.name !== investigatorName) return inv;
              const newLuck = Math.max(0, inv.attributes.Luck.current - luckSpent);
              return {
                  ...inv,
                  attributes: {
                      ...inv.attributes,
                      Luck: { ...inv.attributes.Luck, current: newLuck }
                  }
              };
          })
      }));
      addMessage(MessageSender.SYSTEM, `🍀 ${investigatorName} spent ${luckSpent} Luck to turn failure into success.`);
  }, []);

  const handleDigitalRoll = async (result: number, type: string) => {
    setSuggestedActions([]);
    addMessage(MessageSender.SYSTEM, `🎲 ${type}: ${result}`);
    setIsLoading(true);
    try {
      const msg = `[SYSTEM] The investigator rolled a digital ${type}. Result: ${result}.`;
      const response = await apiService.sendMessage(msg, gameState.settings);
      await processAiResponse(response);
    } catch (error) {
      addMessage(MessageSender.SYSTEM, "Connection error.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualRoll = async (name: string, result: number, skillName?: string, skillValue?: number) => {
    setSuggestedActions([]);

    const textToDisplay = `🎲 ${name}${skillName ? ` (${skillName})` : ''}: ${result} ${skillName ? `/ ${skillValue}` : ''}`;
    let msgToBackend = '';

    if (skillName && skillValue !== undefined) {
      let verdict = "FAILURE";
      if (result === 1)                                        verdict = "CRITICAL SUCCESS";
      else if (result === 100 || (skillValue < 50 && result >= 96)) verdict = "FUMBLE (CRITICAL FAILURE)";
      else if (result <= Math.floor(skillValue / 5))           verdict = "EXTREME SUCCESS";
      else if (result <= Math.floor(skillValue / 2))           verdict = "HARD SUCCESS";
      else if (result <= skillValue)                           verdict = "REGULAR SUCCESS";

      const lastKeeperMsg = [...messages].reverse().find(m => m.sender === MessageSender.KEEPER);
      const pendingContext = lastKeeperMsg
        ? `\n\nSCENE CONTEXT (the action being resolved):\n"${lastKeeperMsg.content.slice(0, 400)}"`
        : '';

      msgToBackend = `[SYSTEM MESSAGE — DICE RESULT — READ THIS FIRST]:
▶ Investigator: ${name}
▶ Skill checked: ${skillName} (target value: ${skillValue})
▶ Dice roll: ${result}
▶ VERDICT: *** ${verdict} ***
${pendingContext}

CRITICAL INSTRUCTION FOR KEEPER: This roll resolves the action described in SCENE CONTEXT above.
Narrate ONLY the direct outcome of THIS roll as a ${verdict} — 1 to 3 sentences.
DO NOT invent a new scene or location. DO NOT repeat prior atmosphere or description.
Continue from exactly where the scene context left off.`;
    } else {
      msgToBackend = `[SYSTEM MESSAGE]: Investigator ${name} performed a raw dice roll. Result: ${result}.`;
    }

    addMessage(MessageSender.SYSTEM, textToDisplay);
    setIsLoading(true);
    try {
      const response = await apiService.sendMessage(msgToBackend, gameState.settings);
      await processAiResponse(response);
    } catch (error) {
      addMessage(MessageSender.SYSTEM, "Connection error.");
    } finally {
      setIsLoading(false);
    }
  };

  const hasRollSuggestion = suggestedActions.some(s =>
    s.toLowerCase().includes('roll') || s.toLowerCase().includes('check') || s.toLowerCase().includes('d100')
  );

  // ── Auto-skill detection: either from clicked Roll action or from last message ──
  const [autoSkill, setAutoSkill] = useState<{ investigatorName: string; skillName: string } | null>(null);

  // Reset autoSkill when a new Keeper message arrives
  const lastKeeperMsg = [...messages].filter(m => m.sender === MessageSender.KEEPER).pop();
  const lastKeeperMsgId = lastKeeperMsg?.id;
  React.useEffect(() => { setAutoSkill(null); }, [lastKeeperMsgId]);

  // Helper: given a skill name string from the LLM, find best investigator + matched skill
  const resolveSkillTarget = (rawSkill: string): { investigatorName: string; skillName: string } | null => {
    if (!rawSkill || gameState.investigators.length === 0) return null;
    const search = rawSkill.toLowerCase().trim();
    let best: { investigatorName: string; skillName: string; value: number } | null = null;
    for (const inv of gameState.investigators) {
      const match = inv.skills.find(s =>
        s.name.toLowerCase() === search ||
        s.name.toLowerCase().includes(search) ||
        search.includes(s.name.toLowerCase())
      );
      if (match && (!best || match.value > best.value)) {
        best = { investigatorName: inv.name, skillName: match.name, value: match.value };
      }
    }
    return best ? { investigatorName: best.investigatorName, skillName: best.skillName } : null;
  };

  // Passive detection from last message (existing behaviour)
  const detectedSkillTarget: { investigatorName: string; skillName: string } | null = (() => {
    if (autoSkill) return autoSkill; // explicit click wins
    if (!lastKeeperMsg || gameState.investigators.length === 0) return null;
    const textToCheck = (lastKeeperMsg.content + ' ' + suggestedActions.join(' ')).toLowerCase();
    let targetInvestigator = gameState.investigators[0];
    const foundInv = gameState.investigators.find(inv => textToCheck.includes(inv.name.toLowerCase()));
    if (foundInv) targetInvestigator = foundInv;
    const foundSkill = targetInvestigator.skills.find(s =>
      s.name.length > 2 && textToCheck.includes(s.name.toLowerCase())
    );
    return foundSkill ? { investigatorName: targetInvestigator.name, skillName: foundSkill.name } : null;
  })();

  
  if (!authToken) {
    return <AuthScreen onAuthenticated={setAuthToken} />;
  }
  
  if (gameState.phase === 'setup') {
    return <SetupScreen onStart={handleStartGame} isLoading={isLoading} />;
  }

  
  return (
    <div className="flex h-screen bg-[#0a0a0a] text-gray-200 overflow-hidden font-sans">
      <div className={`fixed inset-0 z-20 transform ${showSheet ? 'translate-x-0' : '-translate-x-full'} md:relative md:translate-x-0 md:w-1/3 lg:w-1/4 transition-transform duration-300 bg-black border-r border-gray-800`}>
        <div className="h-full flex flex-col p-2 bg-cthulhu-900">
          <button onClick={() => setShowSheet(false)} className="md:hidden absolute top-4 right-4 z-30 text-white font-bold bg-black/50 px-2 rounded">✕</button>
          <div className="flex-1 min-h-0">
            {gameState.investigators.length > 0 && <CharacterSheet investigators={gameState.investigators} />}
          </div>
          <div className={`mt-2 shrink-0 transition-all duration-500 ${hasRollSuggestion ? 'scale-105 ring-2 ring-cthulhu-blood/30 rounded' : ''}`}>
            <DiceRoller
              investigators={gameState.investigators}
              onRoll={handleDigitalRoll}
              onManualSubmit={handleManualRoll}
              onUseLuck={handleUseLuck} 
              forceActive={hasRollSuggestion}
              autoSelectedSkill={detectedSkillTarget}
            />
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col h-full relative">
        <div className="h-16 bg-cthulhu-900 border-b border-gray-800 flex items-center justify-between px-4 shadow-md z-10">
          <div className="font-serif text-xl text-gray-300 tracking-wider">
            Call of Cthulhu
            {gameState.scenarioTitle && gameState.scenarioTitle !== 'Loading...' && (
              <span className="ml-3 text-xs text-gray-600 font-sans normal-case tracking-normal">
                — {gameState.scenarioTitle}
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <button onClick={() => setShowSettings(true)} className="px-2 py-1 bg-cthulhu-800 hover:bg-cthulhu-700 rounded text-sm border border-gray-600 transition-colors">
              ⚙️
            </button>
            <button onClick={() => setShowSheet(!showSheet)} className="md:hidden px-3 py-1 bg-cthulhu-700 rounded text-sm border border-gray-600">Sheet</button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden relative bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-900 to-black">
          <div className="absolute inset-0 flex flex-col p-4 max-w-4xl mx-auto w-full">
            <ChatInterface messages={messages} isLoading={isLoading} />
          </div>
        </div>

        <div className="bg-cthulhu-900 p-4 border-t border-gray-800">
          <div className="max-w-4xl mx-auto flex flex-col gap-3">
            {suggestedActions.length > 0 && (
              <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                {suggestedActions.map((action, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      // Check if this action requires a roll
                      const rollMatch = action.match(/→\s*Roll\s+(.+)$/i);
                      if (rollMatch) {
                        const skillRaw = rollMatch[1].trim();
                        const resolved = resolveSkillTarget(skillRaw);
                        if (resolved) {
                          setAutoSkill(resolved);
                          // Send the action text minus the roll instruction — just the narrative intent
                          const actionText = action.replace(/→\s*Roll\s+.+$/i, '').trim();
                          handleSend(undefined, actionText || action);
                          return;
                        }
                      }
                      handleSend(undefined, action);
                    }}
                    disabled={isLoading}
                    className="bg-gray-800 hover:bg-cthulhu-700 disabled:opacity-50 text-xs md:text-sm text-gray-300 px-3 py-1 rounded-full border border-gray-600 transition-colors"
                  >
                    {action}
                  </button>
                ))}
              </div>
            )}
            <div className="w-full relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.ctrlKey && e.key === 'Enter') handleSend(); }}
                disabled={isLoading}
                placeholder={gameState.language === 'ru' ? "Ваши действия... (Ctrl+Enter)" : "Your actions... (Ctrl+Enter)"}
                className="w-full bg-black border border-gray-600 text-gray-200 p-3 pr-12 rounded shadow-inner focus:border-cthulhu-blood focus:ring-1 focus:ring-cthulhu-blood outline-none transition-all h-14 resize-none"
              />
              <button
                onClick={(e) => handleSend(e as any)}
                disabled={isLoading || !input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white disabled:opacity-30"
              >
                ➤
              </button>
            </div>
          </div>
        </div>
      </div>

      <SettingsModal
        settings={gameState.settings}
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onUpdate={(newSettings) => setGameState(prev => ({ ...prev, settings: newSettings }))}
      />
    </div>
  );
}

export default App;
