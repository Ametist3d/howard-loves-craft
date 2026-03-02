import React, { useState, useEffect } from 'react';
import { Language } from '../types';
import { DATA_EN, DATA_RU } from '../data/names';
import { apiService } from '../services/apiService';

interface InvestigatorConfig {
  id: number;
  name: string;
  occupation: string;
  background: string;
}

interface PrebuiltScenario {
  id: string;
  title: string;
  content: string;
}

interface Props {
  onStart: (config: {
    investigators: InvestigatorConfig[];
    scenario: 'prebuilt' | 'random' | 'custom';
    customPrompt: string;
    themes?: string[];
    language: Language;
    prebuiltScenario?: PrebuiltScenario | null;
  }) => void;
  isLoading: boolean;
}

const THEMES = [
  { id: 'classic',       labelRu: 'Классика',        labelEn: 'Classic' },
  { id: 'expedition',    labelRu: 'Экспедиции',       labelEn: 'Expeditions' },
  { id: 'urban',         labelRu: 'Урбанистика',      labelEn: 'Urban Horror' },
  { id: 'rural',         labelRu: 'Глубинка',         labelEn: 'Rural' },
  { id: 'mystic',        labelRu: 'Мистика',          labelEn: 'Mystic' },
  { id: 'deep_space',    labelRu: 'Глубокий космос',  labelEn: 'Deep Space' },
  { id: 'tech',          labelRu: 'Техно',            labelEn: 'Tech' },
  { id: 'sea',           labelRu: 'Море',             labelEn: 'Sea' },
  { id: 'arctic',        labelRu: 'Арктика',          labelEn: 'Arctic' },
  { id: 'dream',         labelRu: 'Сновидения',       labelEn: 'Dreamlands' },
  { id: 'subterranean',  labelRu: 'Подземелье',       labelEn: 'Subterranean' },
  { id: 'war',           labelRu: 'Война',            labelEn: 'Wartime' },
  { id: 'hospital',      labelRu: 'Больница',         labelEn: 'Hospital' },
  { id: 'ancient',       labelRu: 'Древний мир',      labelEn: 'Ancient World' },
  { id: 'suburban',      labelRu: 'Пригород',         labelEn: 'Suburban' },
  { id: 'post_collapse', labelRu: 'После краха',      labelEn: 'Post-Collapse' },
  { id: 'cult',          labelRu: 'Культы',           labelEn: 'Cults' },
];

export const SetupScreen: React.FC<Props> = ({ onStart, isLoading }) => {
  const [investigators, setInvestigators] = useState<InvestigatorConfig[]>([]);
  const [language, setLanguage] = useState<Language>('ru');
  const [newName, setNewName] = useState('');
  const [newOccupation, setNewOccupation] = useState('');
  const [newBackground, setNewBackground] = useState('');
  const [scenario, setScenario] = useState<'prebuilt' | 'random' | 'custom'>('prebuilt');
  const [prebuiltScenarios, setPrebuiltScenarios] = useState<PrebuiltScenario[]>([]);
  const [selectedPrebuilt, setSelectedPrebuilt] = useState<PrebuiltScenario | null>(null);
  const [prebuiltLoading, setPrebuiltLoading] = useState(false);
  const [selectedThemes, setSelectedThemes] = useState<string[]>([]);
  const [customPrompt, setCustomPrompt] = useState('');

  useEffect(() => {
    setPrebuiltLoading(true);
    apiService.getScenarios()
      .then(setPrebuiltScenarios)
      .finally(() => setPrebuiltLoading(false));
  }, []);

  const addInvestigator = () => {
    if (!newName || !newOccupation) return;
    setInvestigators([...investigators, { id: Date.now(), name: newName, occupation: newOccupation, background: newBackground }]);
    setNewName(''); setNewOccupation(''); setNewBackground('');
  };

  const handleRandomFill = () => {
    const data = language === 'ru' ? DATA_RU : DATA_EN;
    setNewName(`${data.firstNames[Math.floor(Math.random() * data.firstNames.length)]} ${data.lastNames[Math.floor(Math.random() * data.lastNames.length)]}`);
    setNewOccupation(data.occupations[Math.floor(Math.random() * data.occupations.length)]);
    setNewBackground(data.backgrounds[Math.floor(Math.random() * data.backgrounds.length)]);
  };

  const toggleTheme = (themeId: string) => {
    if (selectedThemes.includes(themeId)) setSelectedThemes(prev => prev.filter(t => t !== themeId));
    else if (selectedThemes.length < 3) setSelectedThemes(prev => [...prev, themeId]);
  };

  const canStart = investigators.length > 0 && (
    scenario === 'random' ||
    scenario === 'custom' ||
    (scenario === 'prebuilt' && selectedPrebuilt !== null)
  );

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-full text-white bg-black">
        <div className="w-16 h-16 relative mb-8 flex items-center justify-center">
          <div className="absolute inset-0 bg-cthulhu-blood rounded-full opacity-20 animate-ping"></div>
          <div className="w-3 h-3 bg-cthulhu-blood rounded-full shadow-[0_0_20px_rgba(127,29,29,0.8)] animate-pulse"></div>
        </div>
        <h2 className="text-3xl font-serif text-cthulhu-paper mb-2 tracking-widest">
          {language === 'ru' ? 'РИТУАЛ ПРИЗЫВА...' : 'SUMMONING RITUAL...'}
        </h2>
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-cthulhu-900 text-gray-200 flex flex-col overflow-hidden font-sans relative">

      {/* Language toggle */}
      <div className="absolute top-4 right-4 flex bg-black/60 rounded-lg p-1 backdrop-blur-md border border-gray-700 z-50">
        <button onClick={() => setLanguage('ru')} className={`px-3 py-1 text-xs font-bold rounded-md ${language === 'ru' ? 'bg-cthulhu-blood text-white' : 'text-gray-400'}`}>RU</button>
        <button onClick={() => setLanguage('en')} className={`px-3 py-1 text-xs font-bold rounded-md ${language === 'en' ? 'bg-cthulhu-blood text-white' : 'text-gray-400'}`}>EN</button>
      </div>

      {/* Title — flex-none, never grows */}
      <div className="flex-none flex flex-col items-center pt-8 pb-4 z-10 text-center">
        <h1 className="text-5xl font-serif text-cthulhu-paper tracking-widest drop-shadow-lg mb-2 border-b-2 border-cthulhu-blood pb-2 px-8">
          Call of Cthulhu
        </h1>
      </div>

      {/* Panels row — flex-1 min-h-0 is the key to scroll working */}
      <div className="flex-1 min-h-0 flex gap-6 px-4 md:px-8 max-w-6xl w-full mx-auto">

        {/* I. SCENARIO */}
        <div className="flex-1 min-w-0 bg-black/60 rounded-lg border border-gray-800 flex flex-col backdrop-blur-md overflow-hidden">
          <div className="flex-none p-3 border-b border-gray-800 bg-cthulhu-900/50 text-center text-cthulhu-paper font-bold font-serif uppercase">
            I. Scenario
          </div>
          <div className="flex-1 min-h-0 flex flex-col p-3 gap-3">
            {/* Tabs */}
            <div className="flex-none grid grid-cols-3 gap-1 bg-black/40 p-1 rounded-lg border border-gray-800">
              {(['prebuilt', 'random', 'custom'] as const).map(tab => (
                <button key={tab} onClick={() => setScenario(tab)}
                  className={`py-2 rounded text-xs font-bold uppercase ${scenario === tab ? 'bg-cthulhu-blood text-white' : 'text-gray-500'}`}>
                  {tab}
                </button>
              ))}
            </div>
            {/* Scroll area */}
            <div className="flex-1 min-h-0 overflow-y-auto pr-1">
              {scenario === 'custom' && (
                <textarea value={customPrompt} onChange={e => setCustomPrompt(e.target.value)}
                  placeholder="Describe the horror..."
                  className="w-full h-full bg-black/30 border border-gray-700 rounded p-3 text-sm text-white resize-none min-h-[120px]" />
              )}
              {scenario === 'random' && (
                <div className="flex flex-col gap-2">
                  {THEMES.map(t => (
                    <button key={t.id} onClick={() => toggleTheme(t.id)}
                      className={`text-left px-3 py-2 rounded border text-xs flex justify-between items-center
                        ${selectedThemes.includes(t.id) ? 'bg-cthulhu-800 border-cthulhu-blood text-cthulhu-paper' : 'bg-black/20 border-gray-800 text-gray-400'}`}>
                      <span>{language === 'ru' ? t.labelRu : t.labelEn}</span>
                      {selectedThemes.includes(t.id) && <span className="text-cthulhu-blood">✓</span>}
                    </button>
                  ))}
                </div>
              )}
              {scenario === 'prebuilt' && (
                <div className="flex flex-col gap-2">
                  {prebuiltLoading && <p className="text-gray-600 text-xs text-center mt-4 animate-pulse">Loading scenarios...</p>}
                  {!prebuiltLoading && prebuiltScenarios.length === 0 && (
                    <p className="text-gray-600 text-xs text-center mt-4">No scenarios found in DB</p>
                  )}
                  {prebuiltScenarios.map(s => (
                    <button key={s.id} onClick={() => setSelectedPrebuilt(s)}
                      className={`text-left px-3 py-2 rounded border text-xs
                        ${selectedPrebuilt?.id === s.id ? 'bg-cthulhu-800 border-cthulhu-blood text-cthulhu-paper' : 'bg-black/20 border-gray-800 text-gray-400'}`}>
                      <div className="font-bold mb-0.5">{s.title}</div>
                      <div className="text-[10px] text-gray-500 line-clamp-2">{s.content.slice(0, 100)}...</div>
                      {selectedPrebuilt?.id === s.id && <span className="text-cthulhu-blood text-[10px]">✓ Selected</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* II. INVESTIGATOR */}
        <div className="flex-1 min-w-0 bg-black/60 rounded-lg border border-gray-800 flex flex-col backdrop-blur-md overflow-hidden">
          <div className="flex-none p-3 border-b border-gray-800 bg-cthulhu-900/50 flex justify-between items-center px-4">
            <span className="text-cthulhu-paper font-bold font-serif uppercase">II. Investigator</span>
            <button onClick={handleRandomFill} className="text-gray-400 hover:text-white text-xs">RANDOM</button>
          </div>
          <div className="flex-1 min-h-0 flex flex-col p-4 gap-3">
            <input type="text" value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name"
              className="flex-none w-full bg-black/30 border border-gray-700 rounded px-3 py-2 text-white text-sm" />
            <input type="text" value={newOccupation} onChange={e => setNewOccupation(e.target.value)} placeholder="Occupation"
              className="flex-none w-full bg-black/30 border border-gray-700 rounded px-3 py-2 text-white text-sm" />
            <textarea value={newBackground} onChange={e => setNewBackground(e.target.value)} placeholder="Backstory..."
              className="flex-1 min-h-0 w-full bg-black/30 border border-gray-700 rounded px-3 py-2 text-white text-xs resize-none" />
            <button onClick={addInvestigator} disabled={!newName}
              className="flex-none w-full py-2 bg-cthulhu-800 hover:bg-cthulhu-700 text-cthulhu-paper font-bold rounded border border-gray-600 text-xs uppercase disabled:opacity-50">
              ADD
            </button>
          </div>
        </div>

        {/* III. PARTY */}
        <div className="flex-1 min-w-0 bg-black/60 rounded-lg border border-gray-800 flex flex-col backdrop-blur-md overflow-hidden">
          <div className="flex-none p-3 border-b border-gray-800 bg-cthulhu-900/50 text-center text-cthulhu-paper font-bold font-serif uppercase">
            III. Party
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-4">
            <ul className="flex flex-col gap-2">
              {investigators.map(inv => (
                <li key={inv.id} className="flex justify-between items-center bg-cthulhu-900/80 p-3 rounded border border-gray-700">
                  <div>
                    <div className="font-bold text-cthulhu-paper text-sm">{inv.name}</div>
                    <div className="text-[10px] text-gray-400 uppercase">{inv.occupation}</div>
                  </div>
                  <button onClick={() => setInvestigators(prev => prev.filter(i => i.id !== inv.id))} className="text-red-500 hover:text-red-400">✕</button>
                </li>
              ))}
            </ul>
          </div>
        </div>

      </div>

      {/* BEGIN — flex-none at bottom */}
      <div className="flex-none py-6 flex flex-col items-center gap-2 z-10">
        <button
          onClick={() => onStart({ investigators, scenario, customPrompt, themes: selectedThemes, language, prebuiltScenario: selectedPrebuilt })}
          disabled={!canStart}
          className="w-full max-w-md bg-cthulhu-paper text-cthulhu-900 font-bold font-serif py-3 rounded border-2 border-cthulhu-paper hover:bg-transparent hover:text-cthulhu-paper transition-all uppercase tracking-widest text-lg disabled:opacity-50"
        >
          {language === 'ru' ? 'НАЧАТЬ' : 'BEGIN'}
        </button>
        {scenario === 'prebuilt' && !selectedPrebuilt && investigators.length > 0 && (
          <p className="text-xs text-gray-600">
            {language === 'ru' ? 'Выберите сценарий выше' : 'Select a scenario above'}
          </p>
        )}
      </div>

    </div>
  );
};