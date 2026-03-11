import React from 'react';
import { AppSettings } from '../types';

interface Props {
  settings: AppSettings;
  isOpen: boolean;
  onClose: () => void;
  onUpdate: (newSettings: AppSettings) => void;
}

export const SettingsModal: React.FC<Props> = ({ settings, isOpen, onClose, onUpdate }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-cthulhu-900 border border-gray-600 rounded-lg p-6 w-80 shadow-2xl font-sans text-gray-200">
        <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
          <h2 className="text-xl font-serif text-cthulhu-paper tracking-widest uppercase">Settings</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
        </div>

        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer group">
            <span className="text-sm font-bold group-hover:text-white transition-colors">Enable RAG</span>
            <input 
              type="checkbox" 
              checked={settings.ragEnabled} 
              onChange={(e) => onUpdate({ ...settings, ragEnabled: e.target.checked })}
              className="w-4 h-4 accent-cthulhu-blood"
            />
          </label>

          <label className="flex flex-col space-y-1">
              <span className="text-sm font-bold flex justify-between">
                <span>RAG Top-K</span>
                <span className="text-cthulhu-blood">{settings.topK}</span>
              </span>
              <input
                type="range" min="1" max="10" step="1"
                value={settings.topK}
                onChange={(e) => onUpdate({ ...settings, topK: parseInt(e.target.value) })}
                className="w-full accent-gray-500"
              />
              <span className="text-[10px] text-gray-500">How many RAG chunks to retrieve per query</span>
            </label>
            
          <label className="flex flex-col space-y-1">
            <span className="text-sm font-bold flex justify-between">
              <span>Context Window</span>
              <span className="text-cthulhu-blood">{(settings.numCtx / 1024).toFixed(0)}k</span>
            </span>
            <input
              type="range" min="4096" max="32768" step="4096"
              value={settings.numCtx}
              onChange={(e) => onUpdate({ ...settings, numCtx: parseInt(e.target.value) })}
              className="w-full accent-gray-500"
            />
            <span className="text-[10px] text-gray-500">4k – 32k tokens. Higher = more story memory, slower</span>
          </label>

          <label className="flex flex-col space-y-1">
            <span className="text-sm font-bold flex justify-between">
              <span>LLM Temperature</span>
              <span className="text-cthulhu-blood">{settings.temperature.toFixed(1)}</span>
            </span>
            <input 
              type="range" min="0" max="1.5" step="0.1"
              value={settings.temperature} 
              onChange={(e) => onUpdate({ ...settings, temperature: parseFloat(e.target.value) })}
              className="w-full accent-gray-500"
            />
            <span className="text-[10px] text-gray-500">Higher = More creative/unpredictable</span>
          </label>
        </div>
      </div>
    </div>
  );
};