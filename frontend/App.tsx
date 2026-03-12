import React, { useState, useCallback } from 'react';
import { SetupScreen } from './components/SetupScreen';
import { ChatInterface } from './components/ChatInterface';
import { CharacterSheet } from './components/CharacterSheet';
import { DiceRoller } from './components/DiceRoller';
import { SettingsModal } from './components/SettingsModal';
import { AppSettings } from './types';
import { GameState, ChatMessage, MessageSender, Language, ChatResponse } from './types';
import { apiService } from './services/apiService';
import { SCENARIO_SEEDS } from './data/scenarios';
import { AuthScreen } from './components/AuthScreen';

const INITIAL_STATE: GameState = {
  phase: 'setup',
  investigators: [],
  scenarioTitle: '',
  language: 'ru',
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

  const handleStateUpdate = useCallback((args: any) => {
    if (!args || !args.character_name) return;

    setGameState((prev) => {
      if (prev.investigators.length === 0) return prev;
      const targetName = args.character_name.toLowerCase();

      const index = prev.investigators.findIndex(inv =>
        inv.name.toLowerCase().includes(targetName) || targetName.includes(inv.name.toLowerCase())
      );
      if (index === -1) return prev;

      const updatedInvestigators = [...prev.investigators];
      const inv = { ...updatedInvestigators[index] };
      const attr = { ...inv.attributes };

      if (args.hp_change)      attr.HP.current          = Math.min(attr.HP.max,          Math.max(0, attr.HP.current          + args.hp_change));
      if (args.sanity_change)  attr.Sanity.current      = Math.min(attr.Sanity.max,       Math.max(0, attr.Sanity.current      + args.sanity_change));
      if (args.mp_change)      attr.MagicPoints.current = Math.min(attr.MagicPoints.max,  Math.max(0, attr.MagicPoints.current + args.mp_change));
      if (args.luck_change)    attr.Luck.current        = Math.min(attr.Luck.max,         Math.max(0, attr.Luck.current        + args.luck_change));

      let inventory = [...inv.inventory];
      if (args.inventory_add) inventory.push(args.inventory_add);
      if (args.inventory_remove) {
        const idx = inventory.findIndex(i => i.toLowerCase().includes(args.inventory_remove.toLowerCase()));
        if (idx > -1) inventory.splice(idx, 1);
      }

      let updates: string[] = [];
      if (args.hp_change)     updates.push(`HP ${args.hp_change > 0 ? '+' : ''}${args.hp_change}`);
      if (args.sanity_change) updates.push(`SAN ${args.sanity_change > 0 ? '+' : ''}${args.sanity_change}`);
      if (updates.length > 0) {
        addMessage(MessageSender.SYSTEM, `[${inv.name}: ${updates.join(', ')}]`);
      }

      updatedInvestigators[index] = { ...inv, attributes: attr, inventory };
      return { ...prev, investigators: updatedInvestigators };
    });
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
    if (response.state_updates) handleStateUpdate(response.state_updates);

    // 2. Poll for image in background — doesn't block anything
    if (response.generation_id) {
      const imageUrl = await apiService.pollImageStatus(response.generation_id);
      setMessages((prev) => prev.map((m) =>
        m.id === msgId
          ? { ...m, image: imageUrl || undefined, imageGenerating: false }
          : m
      ));
    }
  };

  const handleStartGame = async (config: {
    investigators: any[];
    scenario: 'prebuilt' | 'random' | 'custom';
    customPrompt: string;
    themes?: string[];
    language: Language;
    prebuiltScenario?: { id: string; title: string; content: string } | null;
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
        if (result.state_updates) handleStateUpdate(result.state_updates);
        // Poll for image if one was queued
        if (result.generation_id) {
          const imageUrl = await apiService.pollImageStatus(result.generation_id);
          setMessages(prev => prev.map(m =>
            m.id === msgId ? { ...m, image: imageUrl || undefined, imageGenerating: false } : m
          ));
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

  let detectedSkillTarget: { investigatorName: string; skillName: string } | null = null;
  const lastMessage = messages[messages.length - 1];

  if (lastMessage && lastMessage.sender === MessageSender.KEEPER && gameState.investigators.length > 0) {
    const textToCheck = (lastMessage.content + ' ' + suggestedActions.join(' ')).toLowerCase();

    let targetInvestigator = gameState.investigators[0];
    const foundInv = gameState.investigators.find(inv => textToCheck.includes(inv.name.toLowerCase()));
    if (foundInv) targetInvestigator = foundInv;

    if (targetInvestigator) {
      // Search against the actual character's skill list — catches custom skills too
      const foundSkill = targetInvestigator.skills.find(s =>
        s.name.length > 2 && textToCheck.includes(s.name.toLowerCase())
      );
      if (foundSkill) {
        detectedSkillTarget = { investigatorName: targetInvestigator.name, skillName: foundSkill.name };
      }
    }
  }

  
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
                    onClick={() => handleSend(undefined, action)}
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
