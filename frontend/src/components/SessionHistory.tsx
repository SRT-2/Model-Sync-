import React, { useState } from 'react';
import {
  Check,
  CheckCircle2,
  Clock,
  Code2,
  Copy,
  Cpu,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { RoutingCategory } from '../types';

export const SessionHistory: React.FC = () => {
  const { sessionHistory, clearSessionHistory } = useApp();
  const [filterCategory, setFilterCategory] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredItems = sessionHistory.filter((item) => {
    if (filterCategory !== 'ALL' && item.category !== filterCategory) {
      return false;
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        item.prompt.toLowerCase().includes(q) ||
        item.response.toLowerCase().includes(q) ||
        (item.model && item.model.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {}
  };

  const renderBadge = (category?: RoutingCategory) => {
    if (!category) return null;
    if (category === 'CODING') {
      return (
        <span className="routing-badge badge-coding">
          <Code2 size={12} />
          <span>CODING</span>
        </span>
      );
    }
    if (category === 'UNIT_TESTING') {
      return (
        <span className="routing-badge badge-testing">
          <CheckCircle2 size={12} />
          <span>TESTING</span>
        </span>
      );
    }
    if (category === 'REASONING') {
      return (
        <span className="routing-badge badge-reasoning">
          <Cpu size={12} />
          <span>REASONING</span>
        </span>
      );
    }
    return null;
  };

  return (
    <div className="history-container">
      {/* Header */}
      <div className="page-header flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Clock className="text-blue-500" size={22} />
            <h1 className="page-title">Session Request History</h1>
          </div>
          <p className="page-subtitle">
            Log of chat requests, classified categories, downstream targets, and latency in the current browser session.
          </p>
        </div>

        {sessionHistory.length > 0 && (
          <button
            type="button"
            onClick={clearSessionHistory}
            className="btn-secondary text-xs text-rose-400 hover:text-rose-300 self-start sm:self-auto"
            aria-label="Clear all session history"
          >
            <Trash2 size={14} />
            <span>Clear Session Log</span>
          </button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 my-4">
        {/* Category Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          {(['ALL', 'CODING', 'UNIT_TESTING', 'REASONING'] as const).map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setFilterCategory(cat)}
              className={`filter-pill ${filterCategory === cat ? 'active' : ''}`}
              aria-label={`Filter by ${cat}`}
            >
              {cat === 'ALL' ? 'All Requests' : cat.replace('_', ' ')}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative min-w-[220px]">
          <Search size={14} className="absolute left-3 top-2.5 text-zinc-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search prompts or output..."
            className="input-field pl-8 text-xs py-1.5"
            aria-label="Search prompt or response"
          />
        </div>
      </div>

      {/* History Items List */}
      {filteredItems.length === 0 ? (
        <div className="empty-history-card">
          <Clock size={32} className="text-zinc-500 mb-2" />
          <p className="text-sm text-zinc-300 font-semibold">No requests match your criteria</p>
          <span className="text-xs text-zinc-400 mt-1">
            {sessionHistory.length === 0
              ? 'Requests sent in the Chat Playground will be tracked here automatically.'
              : 'Try clearing your search query or switching category filters.'}
          </span>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredItems.map((item) => (
            <div key={item.id} className="history-card">
              <div className="history-card-header">
                <div className="flex items-center gap-2 flex-wrap">
                  {renderBadge(item.category)}
                  {item.model && (
                    <span className="model-target-pill">
                      <Sparkles size={11} />
                      <span>{item.model}</span>
                    </span>
                  )}
                  {item.durationMs !== undefined && (
                    <span className="duration-pill">
                      {(item.durationMs / 1000).toFixed(2)}s
                    </span>
                  )}
                  {item.tokens !== undefined && (
                    <span className="duration-pill">
                      {item.tokens} tokens
                    </span>
                  )}
                </div>

                <span className="text-xs text-zinc-400 font-mono">
                  {new Date(item.timestamp).toLocaleTimeString()}
                </span>
              </div>

              {/* Prompt Section */}
              <div className="history-section mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="history-section-title">Prompt</span>
                  <button
                    type="button"
                    onClick={() => handleCopy(`p-${item.id}`, item.prompt)}
                    className="action-btn text-xs"
                    aria-label="Copy prompt"
                  >
                    {copiedId === `p-${item.id}` ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                    <span>{copiedId === `p-${item.id}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <div className="history-prompt-box">
                  {item.prompt}
                </div>
              </div>

              {/* Response Section */}
              <div className="history-section">
                <div className="flex items-center justify-between mb-1">
                  <span className="history-section-title">Response</span>
                  <button
                    type="button"
                    onClick={() => handleCopy(`r-${item.id}`, item.response)}
                    className="action-btn text-xs"
                    aria-label="Copy response"
                  >
                    {copiedId === `r-${item.id}` ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                    <span>{copiedId === `r-${item.id}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <div className="history-response-box">
                  {item.response || '(Empty response)'}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
