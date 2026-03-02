import React, { useState } from 'react';
import { Investigator } from '../types';

interface Props {
  investigators: Investigator[];
}

const STAT_NAMES: Record<string, { en: string; ru: string }> = {
  STR: { en: 'STR', ru: 'СИЛ' },
  CON: { en: 'CON', ru: 'ВЫН' },
  SIZ: { en: 'SIZ', ru: 'ТЕЛ' },
  DEX: { en: 'DEX', ru: 'ЛВК' },
  APP: { en: 'APP', ru: 'НАР' },
  INT: { en: 'INT', ru: 'ИНТ' },
  POW: { en: 'POW', ru: 'МОЩ' },
  EDU: { en: 'EDU', ru: 'ОБР' },
};

export const CharacterSheet: React.FC<Props> = ({ investigators }) => {
  const [activeIndex, setActiveIndex] = useState(0);
  
  if (!investigators || investigators.length === 0) return null;

  const data = investigators[activeIndex];
  if (!data.attributes || !data.characteristics) return null;

  const getSuccessLevels = (val: number) => ({
    normal: val,
    hard: Math.floor(val / 2),
    extreme: Math.floor(val / 5)
  });

  const handlePrint = () => {
    const win = window.open('', '_blank');
    if (!win) return;
    const renderStatBlock = (code: string, val: number) => {
        const lvls = getSuccessLevels(val);
        const label = STAT_NAMES[code].ru;
        return `
        <div class="border border-black flex flex-col text-center">
           <div class="bg-gray-200 font-bold text-[12px] border-b border-black py-1 uppercase">${label}</div>
           <div class="flex flex-1">
             <div class="flex-1 border-r border-black flex items-center justify-center font-bold text-lg">${lvls.normal}</div>
             <div class="flex-1 flex flex-col">
                <div class="flex-1 border-b border-black flex items-center justify-center text-xs">${lvls.hard}</div>
                <div class="flex-1 flex items-center justify-center text-xs">${lvls.extreme}</div>
             </div>
           </div>
        </div>`;
    };

    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>Investigator Dossier: ${data.name}</title>
          <script src="https://cdn.tailwindcss.com"></script>
          <link href="https://fonts.googleapis.com/css2?family=Special+Elite&family=Courier+Prime&display=swap" rel="stylesheet">
          <style>
            body { background-color: #f3f3f3; -webkit-print-color-adjust: exact; }
            .a4-page { width: 210mm; margin: 0 auto; background-color: #fffdf5; padding: 15mm; border: 1px solid #ddd; font-family: 'Special Elite', monospace; color: #1a1a1a; }
            .stamp { border: 3px double #8b0000; color: #8b0000; padding: 5px 10px; transform: rotate(-10deg); display: inline-block; font-weight: bold; text-transform: uppercase; }
            .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
            @media print { body { background: none; } .a4-page { border: none; margin: 0; width: 100%; } }
          </style>
        </head>
        <body>
          <div class="a4-page">
            <div class="flex justify-between items-start border-b-4 border-black pb-4 mb-6">
              <div><h1 class="text-4xl font-bold uppercase tracking-widest">Investigator Sheet</h1><p class="text-sm mt-1">CoC 7th Ed</p></div>
              <div class="stamp">CONFIDENTIAL</div>
            </div>
            <div class="flex gap-6 mb-6">
              <div class="w-32 h-40 border-2 border-black bg-gray-200 flex-shrink-0 overflow-hidden relative">
                 ${data.avatarUrl ? `<img src="${data.avatarUrl}" class="w-full h-full object-cover grayscale contrast-125" />` : '<div class="flex items-center justify-center h-full text-xs text-center">PHOTO MISSING</div>'}
              </div>
              <div class="flex-1 grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                 <div class="border-b border-dotted border-gray-500"><strong class="block text-xs uppercase text-gray-600">Name</strong>${data.name}</div>
                 <div class="border-b border-dotted border-gray-500"><strong class="block text-xs uppercase text-gray-600">Occupation</strong>${data.occupation}</div>
                 <div class="border-b border-dotted border-gray-500"><strong class="block text-xs uppercase text-gray-600">Age</strong>${data.age}</div>
                 <div class="border-b border-dotted border-gray-500"><strong class="block text-xs uppercase text-gray-600">Residence</strong>${data.residence}</div>
              </div>
            </div>
            <div class="grid grid-cols-4 gap-4 mb-6">
               <div class="border-2 border-black p-2 text-center"><div class="text-xs font-bold uppercase">HP</div><div class="text-2xl font-bold">${data.attributes.HP.current} / ${data.attributes.HP.max}</div></div>
               <div class="border-2 border-black p-2 text-center"><div class="text-xs font-bold uppercase">Sanity</div><div class="text-2xl font-bold">${data.attributes.Sanity.current} / ${data.attributes.Sanity.max}</div></div>
               <div class="border-2 border-black p-2 text-center"><div class="text-xs font-bold uppercase">MP</div><div class="text-2xl font-bold">${data.attributes.MagicPoints.current} / ${data.attributes.MagicPoints.max}</div></div>
               <div class="border-2 border-black p-2 text-center"><div class="text-xs font-bold uppercase">Luck</div><div class="text-2xl font-bold">${data.attributes.Luck.current}</div></div>
            </div>
            <div class="mb-6"><h3 class="bg-black text-white px-2 py-1 font-bold uppercase text-sm mb-2">Characteristics</h3>
               <div class="stat-grid">
                  ${renderStatBlock('STR', data.characteristics.STR)} ${renderStatBlock('CON', data.characteristics.CON)}
                  ${renderStatBlock('SIZ', data.characteristics.SIZ)} ${renderStatBlock('DEX', data.characteristics.DEX)}
                  ${renderStatBlock('APP', data.characteristics.APP)} ${renderStatBlock('INT', data.characteristics.INT)}
                  ${renderStatBlock('POW', data.characteristics.POW)} ${renderStatBlock('EDU', data.characteristics.EDU)}
               </div>
            </div>
            <div class="grid grid-cols-2 gap-8 h-auto">
              <div><h3 class="bg-black text-white px-2 py-1 font-bold uppercase text-sm mb-2">Skills</h3><div class="text-xs space-y-1">
                   ${data.skills.map(s => { const lvls = getSuccessLevels(s.value); return `<div class="flex justify-between border-b border-dotted border-gray-400 px-1"><span class="truncate pr-1">${s.name}</span><span class="w-24 flex justify-between font-mono shrink-0"><span class="w-8 text-center">${lvls.normal}</span><span class="w-8 text-center text-gray-600">${lvls.hard}</span><span class="w-8 text-center text-gray-400">${lvls.extreme}</span></span></div>`; }).join('')}
              </div></div>
              <div class="flex flex-col gap-6">
                 <div><h3 class="bg-black text-white px-2 py-1 font-bold uppercase text-sm mb-2">Inventory</h3><ul class="text-xs list-disc pl-4 space-y-1">${data.inventory.map(item => `<li>${item}</li>`).join('')}</ul></div>
                 <div class="flex-1 border-t-2 border-black pt-2"><h3 class="font-bold uppercase text-sm mb-2">Backstory</h3><p class="text-xs text-justify leading-relaxed">${data.background}</p></div>
              </div>
            </div>
          </div>
        </body>
      </html>`;
    win.document.write(htmlContent);
    win.document.close();
  };

  return (
    <div className="flex flex-col h-full max-h-full">
      <div className="flex overflow-x-auto gap-1 mb-2 pb-1 scrollbar-thin shrink-0">
        {investigators.map((inv, idx) => (
          <button key={idx} onClick={() => setActiveIndex(idx)} className={`px-3 py-1 text-sm font-bold font-serif whitespace-nowrap border-t-2 rounded-t transition-colors ${idx === activeIndex ? 'bg-cthulhu-paper text-cthulhu-900 border-cthulhu-blood' : 'bg-cthulhu-800 text-gray-400 border-transparent hover:bg-cthulhu-700'}`}>{inv.name}</button>
        ))}
      </div>
      <div className="bg-cthulhu-paper text-cthulhu-900 p-4 rounded-b rounded-tr shadow-xl overflow-y-auto flex-1 font-serif border-4 border-double border-cthulhu-900 relative">
        <button onClick={handlePrint} className="absolute top-4 right-4 text-cthulhu-900 hover:text-cthulhu-blood">PRINT</button>
        <h2 className="text-xl font-bold text-center border-b-2 border-cthulhu-900 pb-1 mb-4 uppercase tracking-widest">{data.name}</h2>
        
        <div className="flex gap-4 mb-6">
           <div className="w-24 h-24 bg-gray-300 border border-gray-600 flex-shrink-0">{data.avatarUrl ? <img src={data.avatarUrl} alt={data.name} className="w-full h-full object-cover grayscale contrast-125" /> : <div className="w-full h-full flex items-center justify-center text-xs">NO PHOTO</div>}</div>
           <div className="flex-1 grid grid-cols-1 gap-1 text-sm">
            <div className="flex justify-between border-b border-gray-400"><span className="font-bold">Occup:</span> <span>{data.occupation}</span></div>
            <div className="flex justify-between border-b border-gray-400"><span className="font-bold">Age:</span> <span>{data.age}</span></div>
            <div className="flex justify-between border-b border-gray-400"><span className="font-bold">Home:</span> <span>{data.residence}</span></div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 mb-4">
          <div className="bg-gray-200 p-1 border border-gray-400"><div className="text-[10px] font-bold text-center uppercase">HP</div><div className="text-center text-lg font-bold text-red-800">{data.attributes.HP.current}/{data.attributes.HP.max}</div></div>
          <div className="bg-gray-200 p-1 border border-gray-400"><div className="text-[10px] font-bold text-center uppercase">SAN</div><div className="text-center text-lg font-bold text-blue-900">{data.attributes.Sanity.current}/{data.attributes.Sanity.max}</div></div>
          <div className="bg-gray-200 p-1 border border-gray-400"><div className="text-[10px] font-bold text-center uppercase">MP</div><div className="text-center text-lg font-bold text-purple-900">{data.attributes.MagicPoints.current}/{data.attributes.MagicPoints.max}</div></div>
          <div className="bg-gray-200 p-1 border border-gray-400"><div className="text-[10px] font-bold text-center uppercase">LUCK</div><div className="text-center text-lg font-bold text-green-900">{data.attributes.Luck.current}</div></div>
        </div>
        <div className="grid grid-cols-4 gap-2 mb-4 text-center">
          {[['STR', data.characteristics.STR], ['CON', data.characteristics.CON], ['SIZ', data.characteristics.SIZ], ['DEX', data.characteristics.DEX], ['APP', data.characteristics.APP], ['INT', data.characteristics.INT], ['POW', data.characteristics.POW], ['EDU', data.characteristics.EDU]].map(([code, val]) => (
            <div key={code as string} className="bg-black/5 p-1 border border-gray-400 flex flex-col items-center"><div className="font-bold text-[10px] uppercase">{code}</div><div className="font-bold">{val}</div></div>
          ))}
        </div>
        <h3 className="font-bold text-md border-b border-gray-800 mb-2">Skills</h3>
        <div className="grid grid-cols-1 gap-y-1 mb-4 text-xs">
          {data.skills.map((skill, idx) => (<div key={idx} className="flex justify-between border-b border-gray-300 border-dotted"><span className="truncate">{skill.name}</span><span className="font-mono font-bold">{skill.value}</span></div>))}
        </div>
        <h3 className="font-bold text-md border-b border-gray-800 mb-2">Backstory</h3>
        <div className="p-2 bg-yellow-100 border border-yellow-300 text-xs italic text-justify">{data.background}</div>
      </div>
    </div>
  );
};