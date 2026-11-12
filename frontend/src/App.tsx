import React, { useEffect, useState } from 'react';
import { Menu, Layers } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { ChatPlayground } from './components/ChatPlayground';
import { RoutingInsights } from './components/RoutingInsights';
import { ConnectionSettings } from './components/ConnectionSettings';
import { SessionHistory } from './components/SessionHistory';
import { useApp } from './context/AppContext';

export const AppContent: React.FC = () => {
  const { activeTab } = useApp();
  const [darkMode, setDarkMode] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    }
  }, [darkMode]);

  return (
    <div className="app-layout">
      {/* Mobile Top Header */}
      <header className="mobile-header md:hidden">
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          className="p-1.5 text-zinc-400 hover:text-zinc-100"
          aria-label="Open sidebar menu"
        >
          <Menu size={20} />
        </button>

        <div className="flex items-center gap-2">
          <Layers className="text-blue-500" size={18} />
          <span className="font-bold text-sm text-zinc-100">ModelSync</span>
          <span className="brand-badge text-[10px]">v2.0</span>
        </div>

        <div className="w-6" />
      </header>

      {/* Sidebar */}
      <Sidebar
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'chat' && <ChatPlayground />}
        {activeTab === 'insights' && <RoutingInsights />}
        {activeTab === 'models' && <ConnectionSettings />}
        {activeTab === 'history' && <SessionHistory />}
      </main>
    </div>
  );
};
