import React from 'react';
import {
  Activity,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  MessageSquare,
  Moon,
  Route,
  Server,
  ShieldAlert,
  Sun,
  XCircle,
} from 'lucide-react';
import { ActiveTab, useApp } from '../context/AppContext';

interface SidebarProps {
  darkMode: boolean;
  setDarkMode: (val: boolean) => void;
  mobileOpen: boolean;
  setMobileOpen: (val: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  darkMode,
  setDarkMode,
  mobileOpen,
  setMobileOpen,
}) => {
  const { activeTab, setActiveTab, connectionStatus, sessionHistory, health, config } = useApp();

  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: number }[] = [
    {
      id: 'chat',
      label: 'Chat Playground',
      icon: <MessageSquare size={17} />,
    },
    {
      id: 'insights',
      label: 'Routing Insights',
      icon: <Route size={17} />,
    },
    {
      id: 'models',
      label: 'Connection & Models',
      icon: <Server size={17} />,
    },
    {
      id: 'history',
      label: 'Session History',
      icon: <Clock size={17} />,
      badge: sessionHistory.length > 0 ? sessionHistory.length : undefined,
    },
  ];

  const renderStatusBadge = () => {
    switch (connectionStatus) {
      case 'online':
        return (
          <div className="status-indicator status-online" title="Backend connected and responding">
            <span className="status-dot"></span>
            <span>Online</span>
          </div>
        );
      case 'unauthorized':
        return (
          <div className="status-indicator status-warning" title="Authentication required">
            <ShieldAlert size={13} />
            <span>Auth Required</span>
          </div>
        );
      case 'checking':
        return (
          <div className="status-indicator status-checking" title="Checking health...">
            <Activity size={13} className="animate-spin" />
            <span>Checking...</span>
          </div>
        );
      default:
        return (
          <div className="status-indicator status-offline" title="Backend unreachable at current URL">
            <XCircle size={13} />
            <span>Offline</span>
          </div>
        );
    }
  };

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="mobile-backdrop md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside className={`sidebar-container ${mobileOpen ? 'open' : ''}`}>
        {/* Brand Header */}
        <div className="sidebar-brand">
          <div className="brand-logo">
            <Layers className="text-blue-500" size={22} />
          </div>
          <div className="brand-text">
            <div className="flex items-center gap-1.5">
              <span className="brand-title">ModelSync</span>
              <span className="brand-badge">v2.0</span>
            </div>
            <span className="brand-subtitle">Smart LLM Router</span>
          </div>
        </div>

        {/* Backend Status Bar */}
        <div className="sidebar-status-box">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-400 font-medium">Backend Status</span>
            {renderStatusBadge()}
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 font-mono truncate">
            {config.baseUrl}
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveTab(item.id);
                  setMobileOpen(false);
                }}
                className={`nav-button ${isActive ? 'active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {item.badge !== undefined && (
                  <span className="nav-badge">{item.badge}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Active Model Info */}
        <div className="sidebar-model-info">
          <div className="flex items-center gap-1.5 text-xs text-zinc-400 font-medium mb-1">
            <Cpu size={14} className="text-blue-400" />
            <span>Virtual Model Target</span>
          </div>
          <div className="font-mono text-xs font-semibold text-zinc-200 truncate">
            {config.selectedModel || 'smart-router'}
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">
            Server-side task classification & forwarding
          </div>
        </div>

        {/* Footer actions */}
        <div className="sidebar-footer">
          <button
            type="button"
            onClick={() => setDarkMode(!darkMode)}
            className="theme-toggle-btn"
            title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Sun size={15} /> : <Moon size={15} />}
            <span className="text-xs">{darkMode ? 'Light Theme' : 'Dark Theme'}</span>
          </button>
        </div>
      </aside>
    </>
  );
};
