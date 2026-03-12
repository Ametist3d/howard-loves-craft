import React, { useState } from 'react';

interface Props {
  onAuthenticated: (token: string) => void;
}

export const AuthScreen: React.FC<Props> = ({ onAuthenticated }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!username || !password) return;
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        setError('Invalid username or password.');
        return;
      }
      const data = await res.json();
      sessionStorage.setItem('keeper_token', data.token);
      onAuthenticated(data.token);
    } catch {
      setError('Cannot reach the backend. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleLogin();
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="w-full max-w-sm px-8 py-10 bg-[#0d0d0d] border border-gray-800 rounded-lg shadow-2xl flex flex-col gap-6">

        {/* Header */}
        <div className="text-center">
          <div className="text-4xl mb-3">🐙</div>
          <h1 className="text-cthulhu-paper font-serif text-2xl tracking-widest uppercase">
            Keeper AI
          </h1>
          <p className="text-gray-600 text-xs mt-1 tracking-wider uppercase">
            Enter your credentials to proceed
          </p>
        </div>

        {/* Inputs */}
        <div className="flex flex-col gap-3">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            onKeyDown={handleKey}
            autoFocus
            className="w-full bg-black border border-gray-700 text-gray-300 text-sm rounded px-3 py-2 outline-none focus:border-cthulhu-blood transition-colors placeholder-gray-700 font-mono"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={handleKey}
            className="w-full bg-black border border-gray-700 text-gray-300 text-sm rounded px-3 py-2 outline-none focus:border-cthulhu-blood transition-colors placeholder-gray-700 font-mono"
          />
        </div>

        {/* Error */}
        {error && (
          <p className="text-red-500 text-xs text-center font-mono">{error}</p>
        )}

        {/* Submit */}
        <button
          onClick={handleLogin}
          disabled={loading || !username || !password}
          className="w-full py-2 bg-cthulhu-blood text-white font-bold rounded uppercase tracking-widest text-sm hover:bg-red-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? 'Entering...' : 'Enter'}
        </button>

        <p className="text-gray-800 text-xs text-center font-mono">
          Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn
        </p>
      </div>
    </div>
  );
};