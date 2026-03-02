import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, MessageSender } from '../types';
import ReactMarkdown from 'react-markdown';
import { apiService } from '../services/apiService';

interface Props {
  messages: ChatMessage[];
  isLoading: boolean;
}

export const ChatInterface: React.FC<Props> = ({ messages, isLoading }) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [showBlueprint, setShowBlueprint] = useState(false);
  const [blueprint, setBlueprint] = useState<any>(null);

  const fetchBlueprint = async () => {
    try {
      const data = await apiService.getScenarioBlueprint();
      setBlueprint(data);
      setShowBlueprint(true);
    } catch (e) {
      console.error('Blueprint fetch failed:', e);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-black/50 backdrop-blur-sm rounded-lg border border-gray-800">
      <div className="flex justify-end mb-2">
        <button
          onClick={fetchBlueprint}
          className="text-xs px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-500 hover:text-white hover:border-gray-500 font-mono"
          title="Debug: View Scenario Blueprint"
        >
          📜 SCRIPT
        </button>
      </div>
      {messages.map((msg) => (
        <div key={msg.id} className={`flex ${msg.sender === MessageSender.USER ? 'justify-end' : 'justify-start'}`}>
          <div className={`max-w-[85%] rounded-lg p-4 shadow-lg overflow-hidden ${
              msg.sender === MessageSender.USER
                ? 'bg-cthulhu-700 text-white border-l-4 border-gray-500'
                : msg.sender === MessageSender.SYSTEM
                ? 'bg-gray-900 text-yellow-500 text-sm font-mono border border-yellow-900'
                : 'bg-[#1a1510] text-gray-300 border-r-4 border-cthulhu-blood font-serif leading-relaxed w-full'
            }`}>
            {msg.sender === MessageSender.KEEPER && <div className="text-xs text-cthulhu-blood font-bold uppercase mb-1 tracking-widest">Keeper</div>}
            {msg.image && (
               <div className="mb-3 rounded overflow-hidden border border-cthulhu-700 relative group">
                  <img src={msg.image} alt="Visualization" className="w-full h-auto object-cover animate-fade-in" />
               </div>
            )}
            <div className="prose prose-invert prose-p:mb-2 max-w-none break-words whitespace-normal text-sm md:text-base">
               <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        </div>
      ))}
      
      {isLoading && (
        <div className="flex justify-start animate-pulse">
          <div className="bg-[#1a1510] p-4 rounded-lg border-r-4 border-cthulhu-blood">
             <span className="text-xs text-gray-500 tracking-widest">THINKING...</span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />

      {showBlueprint && blueprint && (
        <div className="fixed inset-0 z-50 bg-black/95 flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
            <h2 className="text-yellow-600 font-serif text-lg uppercase tracking-widest">
              📜 {blueprint.title || 'Scenario Blueprint'}
            </h2>
            <button onClick={() => setShowBlueprint(false)} className="text-gray-500 hover:text-white text-xl">✕</button>
          </div>
          <div className="flex-1 overflow-y-auto p-6 font-mono text-sm text-green-300 whitespace-pre-wrap">
            {JSON.stringify(blueprint, null, 2)}
          </div>
        </div>
      )}
    </div>
  );
};