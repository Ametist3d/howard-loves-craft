import React, { useState, useEffect } from 'react';
import { Investigator } from '../types';

interface DiceRollerProps {
  investigators: Investigator[];
  onRoll: (result: number, type: 'd100' | 'd20' | 'd10' | 'd8' | 'd6' | 'd4') => void;
  onManualSubmit: (investigatorName: string, result: number, skillName?: string, skillValue?: number, luckSpent?: number) => void;
  onUseLuck: (investigatorName: string, luckSpent: number) => void;
  forceActive?: boolean;
  autoSelectedSkill?: { investigatorName: string, skillName: string } | null;
}

export const DiceRoller: React.FC<DiceRollerProps> = ({ investigators, onRoll, onManualSubmit, onUseLuck, forceActive, autoSelectedSkill }) => {
  const [rolling, setRolling] = useState(false);
  const [lastResult, setLastResult] = useState<number | null>(null);
  const [pendingResult, setPendingResult] = useState<number | null>(null); 
  
  const [selectedInvestigator, setSelectedInvestigator] = useState<string>('');
  const [selectedSkill, setSelectedSkill] = useState<string>('');
  const [manualValue, setManualValue] = useState<string>('');

  useEffect(() => {
    if (investigators.length > 0 && !selectedInvestigator) {
      setSelectedInvestigator(investigators[0].name);
    }
  }, [investigators, selectedInvestigator]);

  useEffect(() => {
      if (autoSelectedSkill && investigators.length > 0 && autoSelectedSkill.investigatorName) {
          const inv = investigators.find(i => i.name.toLowerCase().includes(autoSelectedSkill.investigatorName.toLowerCase()));
          
          if (inv && inv.characteristics && inv.attributes) {
              setSelectedInvestigator(inv.name);
              const chars = inv.characteristics;
              const search = autoSelectedSkill.skillName ? autoSelectedSkill.skillName.toLowerCase() : '';
              
              if (!search) return;

              const skill = inv.skills.find(s => s.name.toLowerCase() === search || s.name.toLowerCase().includes(search));
              if (skill) {
                  setSelectedSkill(`${skill.name}|${skill.value}`);
              } else {
                    if (search.includes('str')) setSelectedSkill(`STR|${chars.STR}`);
                    else if (search.includes('con')) setSelectedSkill(`CON|${chars.CON}`);
                    else if (search.includes('siz')) setSelectedSkill(`SIZ|${chars.SIZ}`);
                    else if (search.includes('dex')) setSelectedSkill(`DEX|${chars.DEX}`);
                    else if (search.includes('app')) setSelectedSkill(`APP|${chars.APP}`);
                    else if (search.includes('int')) setSelectedSkill(`INT|${chars.INT}`);
                    else if (search.includes('pow')) setSelectedSkill(`POW|${chars.POW}`);
                    else if (search.includes('edu')) setSelectedSkill(`EDU|${chars.EDU}`);
                    else if (search.includes('luck')) setSelectedSkill(`Luck|${inv.attributes.Luck.current}`);
                    else if (search.includes('san')) setSelectedSkill(`Sanity|${inv.attributes.Sanity.current}`);
              }
          }
      }
  }, [autoSelectedSkill, investigators]);

  useEffect(() => {
      if (!forceActive) {
          setPendingResult(null);
      }
  }, [forceActive]);

  const getCurrentInvestigator = () => investigators.find(i => i.name === selectedInvestigator);

  const getLuckSave = () => {
      if (!pendingResult || !selectedSkill) return null;
      const inv = getCurrentInvestigator();
      if (!inv) return null;
      const [, valStr] = selectedSkill.split('|');
      const skillValue = parseInt(valStr);
      const luck = inv.attributes?.Luck?.current ?? 0;
      // Already a success — no luck needed
      if (pendingResult <= skillValue) return null;
      const cost = pendingResult - skillValue;
      // Not enough luck to cover the gap
      if (cost > luck) return null;
      return { cost, remainingLuck: luck - cost };
  };

  const roll = (sides: number, type: 'd100' | 'd20' | 'd10' | 'd8' | 'd6' | 'd4') => {
    if (rolling) return;
    setRolling(true);
    
    let count = 0;
    const interval = setInterval(() => {
      setLastResult(Math.floor(Math.random() * sides) + 1);
      count++;
      if (count > 10) {
        clearInterval(interval);
        const final = Math.floor(Math.random() * sides) + 1;
        setLastResult(final);
        setRolling(false);
        
        if (forceActive || selectedSkill) {
            setPendingResult(final);
        } else {
             onRoll(final, type);
        }
      }
    }, 50);
  };

  const confirmRoll = () => {
      const inv = getCurrentInvestigator();
      const result = pendingResult !== null ? pendingResult : parseInt(manualValue);
      
      if (inv && !isNaN(result)) {
          let skillName = undefined;
          let skillValue = undefined;
          
          if (selectedSkill) {
              const [name, valStr] = selectedSkill.split('|');
              skillName = name;
              skillValue = parseInt(valStr);
          }

          onManualSubmit(inv.name, result, skillName, skillValue);
          setManualValue('');
          setPendingResult(null);
          setLastResult(result);
      }
  };

  const useLuck = () => {
      const inv = getCurrentInvestigator();
      const save = getLuckSave();
      if (!inv || !save || !selectedSkill) return;
      const [skillName, valStr] = selectedSkill.split('|');
      const skillValue = parseInt(valStr);
      // Deduct luck in parent state
      onUseLuck(inv.name, save.cost);
      // Submit as a success at exactly the threshold
      onManualSubmit(inv.name, skillValue, skillName, skillValue, save.cost);
      setPendingResult(null);
      setLastResult(skillValue);
  };

  const renderSkillOptions = () => {
      const inv = getCurrentInvestigator();
      if (!inv || !inv.characteristics) return <option>No Character</option>;

      const options = [];
      const chars = inv.characteristics;

      options.push(<option key="str" value={`STR|${chars.STR}`}>STR ({chars.STR})</option>);
      options.push(<option key="con" value={`CON|${chars.CON}`}>CON ({chars.CON})</option>);
      options.push(<option key="siz" value={`SIZ|${chars.SIZ}`}>SIZ ({chars.SIZ})</option>);
      options.push(<option key="dex" value={`DEX|${chars.DEX}`}>DEX ({chars.DEX})</option>);
      options.push(<option key="app" value={`APP|${chars.APP}`}>APP ({chars.APP})</option>);
      options.push(<option key="int" value={`INT|${chars.INT}`}>INT ({chars.INT})</option>);
      options.push(<option key="pow" value={`POW|${chars.POW}`}>POW ({chars.POW})</option>);
      options.push(<option key="edu" value={`EDU|${chars.EDU}`}>EDU ({chars.EDU})</option>);

      if (inv.attributes && inv.attributes.Luck && inv.attributes.Sanity) {
          options.push(<option key="luck" value={`Luck|${inv.attributes.Luck.current}`}>Luck ({inv.attributes.Luck.current})</option>);
          options.push(<option key="san" value={`Sanity|${inv.attributes.Sanity.current}`}>Sanity ({inv.attributes.Sanity.current})</option>);
      }
      options.push(<option disabled key="sep1">──────────</option>);

      if (inv.skills) {
          const sortedSkills = [...inv.skills].sort((a, b) => a.name.localeCompare(b.name));
          sortedSkills.forEach((skill, idx) => {
              options.push(<option key={`s-${idx}`} value={`${skill.name}|${skill.value}`}>{skill.name} ({skill.value})</option>);
          });
      }

      return options;
  };

  const DiceButton = ({ sides, type }: { sides: number, type: 'd100' | 'd20' | 'd10' | 'd8' | 'd6' | 'd4' }) => (
    <button onClick={() => roll(sides, type)} disabled={rolling || pendingResult !== null} className="px-2 py-2 bg-gray-800 text-cthulhu-paper border border-gray-600 rounded hover:bg-cthulhu-blood transition-colors text-xs font-bold disabled:opacity-50 uppercase shadow-sm">
      {type}
    </button>
  );

  return (
    <div className={`bg-cthulhu-900 p-3 rounded border border-cthulhu-700 flex flex-col items-center gap-3 shadow-lg w-full sm:w-auto transition-all duration-300 ${forceActive ? 'ring-2 ring-cthulhu-blood shadow-[0_0_15px_rgba(127,29,29,0.5)]' : ''}`}>
      <div className="text-cthulhu-paper font-serif text-sm font-bold uppercase tracking-wider border-b border-gray-700 w-full text-center pb-1 flex justify-between items-center">
        <span>Dice of Fate</span>
        {forceActive && <span className="text-[10px] text-red-500 animate-pulse">● ROLL</span>}
      </div>
      
      <div className={`text-3xl font-bold text-red-600 h-10 flex items-center justify-center w-full bg-black/30 rounded border border-gray-800 font-serif ${rolling ? 'animate-pulse' : ''}`}>
        {lastResult !== null ? lastResult : '--'}
      </div>

      <div className="grid grid-cols-3 gap-2 w-full">
        <DiceButton sides={100} type="d100" />
        <DiceButton sides={20} type="d20" />
        <DiceButton sides={10} type="d10" />
        <DiceButton sides={8} type="d8" />
        <DiceButton sides={6} type="d6" />
        <DiceButton sides={4} type="d4" />
      </div>

      {investigators.length > 0 && (
        <div className="w-full flex flex-col gap-2 pt-2 border-t border-gray-700 mt-1">
           <div className="text-[10px] text-gray-500 uppercase text-center">Skill Check</div>
           
           {!pendingResult && (
               <div className="flex flex-col gap-2 mb-1">
                <select value={selectedInvestigator} onChange={(e) => setSelectedInvestigator(e.target.value)} className="w-full bg-black border border-gray-600 text-gray-300 text-xs rounded p-1 outline-none">
                    {investigators.map((inv, idx) => (
                    <option key={idx} value={inv.name}>{inv.name}</option>
                    ))}
                </select>

                <select value={selectedSkill} onChange={(e) => setSelectedSkill(e.target.value)} className="w-full bg-black border border-gray-600 text-gray-300 text-xs rounded p-1 outline-none">
                    <option value="">-- Select Skill --</option>
                    {renderSkillOptions()}
                </select>

                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 whitespace-nowrap">Result:</span>
                    <input type="number" value={manualValue} onChange={(e) => setManualValue(e.target.value)} placeholder="00" className="w-full bg-black border border-gray-600 text-gray-300 text-xs rounded p-1 outline-none text-center"/>
                </div>
               </div>
           )}

           {(() => {
              const luckSave = getLuckSave();
              const isSuccess = pendingResult !== null && selectedSkill
                  ? pendingResult <= parseInt(selectedSkill.split('|')[1])
                  : null;
              return (
                  <>
                  {pendingResult !== null && selectedSkill && (
                      <div className={`text-center text-xs font-mono py-1 rounded ${isSuccess ? 'text-green-400' : 'text-red-400'}`}>
                          {isSuccess ? '✓ SUCCESS' : `✗ FAIL — need ≤${selectedSkill.split('|')[1]}, rolled ${pendingResult}`}
                      </div>
                  )}
                  {luckSave && (
                      <button onClick={useLuck}
                          className="w-full py-2 bg-yellow-800 hover:bg-yellow-600 text-yellow-100 text-xs font-bold rounded uppercase tracking-widest transition-all border border-yellow-600">
                          🍀 USE LUCK  (–{luckSave.cost} → {luckSave.remainingLuck} remaining)
                      </button>
                  )}
                  <button onClick={confirmRoll}
                      disabled={(!pendingResult && !manualValue)}
                      className={`w-full py-2 text-sm font-bold rounded transition-all uppercase tracking-widest ${pendingResult || manualValue ? 'bg-cthulhu-blood text-white hover:bg-red-600 shadow-md animate-pulse' : 'bg-gray-700 text-gray-500 cursor-not-allowed'}`}>
                      {pendingResult ? 'CONFIRM' : 'MANUAL SUBMIT'}
                  </button>
                  </>
              );
          })()}
        </div>
      )}
    </div>
  );
};