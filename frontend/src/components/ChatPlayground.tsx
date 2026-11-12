import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  Code2,
  CheckCircle2,
  Cpu,
  Layers,
  Send,
  Sliders,
  Square,
  Trash2,
  Sparkles,
  X,
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { MessageItem } from './MessageItem';

export const ChatPlayground: React.FC = () => {
  const {
    messages,
    sendMessage,
    stopGeneration,
    regenerateLast,
    clearChat,
    isGenerating,
    generationParams,
    setGenerationParams,
    connectionStatus,
    config,
  } = useApp();

  const [inputPrompt, setInputPrompt] = useState('');
  const [showParamsModal, setShowParamsModal] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom of conversation
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputPrompt(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!inputPrompt.trim() || isGenerating) return;
    const prompt = inputPrompt;
    setInputPrompt('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    sendMessage(prompt);
  };

  const samplePrompts = [
    {
      category: 'CODING',
      icon: <Code2 size={16} className="text-blue-400" />,
      label: 'Coding Task',
      prompt: 'Implement a thread-safe LRU Cache with TTL expiration in Python using asyncio and double-linked list.',
    },
    {
      category: 'UNIT_TESTING',
      icon: <CheckCircle2 size={16} className="text-emerald-400" />,
      label: 'Unit Testing Task',
      prompt: 'Write comprehensive pytest test cases including parametrization, boundary conditions, and mock fixtures for a JWT authentication middleware.',
    },
    {
      category: 'REASONING',
      icon: <Cpu size={16} className="text-purple-400" />,
      label: 'Reasoning Task',
      prompt: 'Analyze the architectural and consistency trade-offs between Event Sourcing with CQRS vs traditional 3NF relational transactions in distributed billing systems.',
    },
  ];

  return (
    <div className="chat-playground-container">
      {/* Playground Header */}
      <header className="playground-header">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-zinc-100">Playground</span>
            <span className="model-chip" title="Virtual model target exposed to clients">
              <Layers size={13} className="text-blue-400" />
              <span>{config.selectedModel || 'smart-router'}</span>
            </span>
          </div>
          {connectionStatus === 'offline' && (
            <span className="offline-notice">
              <AlertCircle size={13} />
              <span>Backend unreachable</span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Generation parameters button */}
          <button
            type="button"
            onClick={() => setShowParamsModal(!showParamsModal)}
            className={`btn-secondary ${showParamsModal ? 'active' : ''}`}
            title="Configure inference parameters"
            aria-expanded={showParamsModal}
          >
            <Sliders size={14} />
            <span className="hidden sm:inline">Parameters</span>
          </button>

          {/* Clear conversation button */}
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => setShowClearConfirm(true)}
              className="btn-secondary text-rose-400 hover:text-rose-300"
              title="Clear conversation"
              aria-label="Clear conversation"
            >
              <Trash2 size={14} />
              <span className="hidden sm:inline">Clear</span>
            </button>
          )}
        </div>
      </header>

      {/* Generation Parameters Flyout */}
      {showParamsModal && (
        <div className="params-flyout" role="dialog" aria-label="Inference Parameters">
          <div className="params-header">
            <h4 className="params-title">
              Inference Parameters
            </h4>
            <button
              type="button"
              onClick={() => setShowParamsModal(false)}
              className="close-btn"
              aria-label="Close parameters drawer"
            >
              <X size={15} />
            </button>
          </div>

          <div className="space-y-4">
            {/* Stream toggle */}
            <div className="flex items-center justify-between">
              <div>
                <label htmlFor="stream-toggle" className="text-xs font-semibold text-zinc-200 block">
                  Stream Responses
                </label>
                <span className="text-[11px] text-zinc-400">Stream chunks via Server-Sent Events</span>
              </div>
              <input
                id="stream-toggle"
                type="checkbox"
                checked={generationParams.stream}
                onChange={(e) =>
                  setGenerationParams((prev) => ({ ...prev, stream: e.target.checked }))
                }
                className="toggle-checkbox"
                aria-label="Enable streaming responses"
              />
            </div>

            {/* Temperature */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="temp-slider" className="text-zinc-200 font-medium">Temperature</label>
                <span className="font-mono text-blue-400 font-semibold">{generationParams.temperature ?? 0.7}</span>
              </div>
              <input
                id="temp-slider"
                type="range"
                min="0.0"
                max="1.5"
                step="0.05"
                value={generationParams.temperature ?? 0.7}
                onChange={(e) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    temperature: parseFloat(e.target.value),
                  }))
                }
                aria-label="Temperature slider"
              />
            </div>

            {/* Max Tokens */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="max-tokens-input" className="text-zinc-200 font-medium">Max Tokens</label>
                <span className="font-mono text-zinc-400 font-semibold">{generationParams.max_tokens ?? 2048}</span>
              </div>
              <input
                id="max-tokens-input"
                type="number"
                min="64"
                max="8192"
                step="128"
                value={generationParams.max_tokens ?? 2048}
                onChange={(e) =>
                  setGenerationParams((prev) => ({
                    ...prev,
                    max_tokens: parseInt(e.target.value) || 2048,
                  }))
                }
                className="input-field text-xs py-1.5"
                aria-label="Max tokens input"
              />
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal for Clearing Chat */}
      {showClearConfirm && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="clear-dialog-title">
          <div className="modal-content">
            <h3 id="clear-dialog-title" className="modal-title">Clear Conversation?</h3>
            <p className="modal-desc">
              This will remove all current messages from the active playground. Past completed requests remain safely recorded in Session History.
            </p>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowClearConfirm(false)}
                className="btn-secondary text-xs"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  clearChat();
                  setShowClearConfirm(false);
                }}
                className="btn-danger text-xs"
              >
                Clear Conversation
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Conversation Thread */}
      <div className="chat-messages-area">
        {messages.length === 0 ? (
          <div className="empty-chat-state">
            <div className="empty-state-icon">
              <Sparkles size={32} className="text-blue-400" />
            </div>
            <h2 className="empty-title">
              ModelSync Smart Router
            </h2>
            <p className="empty-subtitle">
              Send your request to the virtual <code>smart-router</code> endpoint. The backend classifier automatically analyzes your prompt and forwards it to the optimal specialized model.
            </p>

            {/* Starter Suggestion Pills */}
            <div className="suggestion-grid">
              {samplePrompts.map((sample, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setInputPrompt(sample.prompt);
                    if (textareaRef.current) {
                      textareaRef.current.focus();
                    }
                  }}
                  className="suggestion-card"
                  aria-label={`Load example prompt for ${sample.label}`}
                >
                  <div className="suggestion-header">
                    {sample.icon}
                    <span className="suggestion-title">{sample.label}</span>
                  </div>
                  <p className="suggestion-prompt-text">
                    {sample.prompt}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, index) => (
              <MessageItem
                key={msg.id}
                message={msg}
                onRegenerate={
                  index === messages.length - 1 && msg.role === 'assistant'
                    ? regenerateLast
                    : undefined
                }
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Chat Input Section */}
      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            rows={1}
            value={inputPrompt}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type your coding, unit-testing, or reasoning prompt... (Enter to send, Shift+Enter for new line)"
            className="chat-textarea"
            disabled={isGenerating}
            aria-label="User prompt input"
          />

          <div className="input-actions-bar">
            <span className="input-hint hidden sm:inline">
              Press <b>Enter ↵</b> to send, <b>Shift+Enter</b> for new line
            </span>

            {isGenerating ? (
              <button
                type="button"
                onClick={stopGeneration}
                className="btn-stop"
                title="Cancel streaming"
                aria-label="Stop generation"
              >
                <Square size={13} className="fill-current" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!inputPrompt.trim() || connectionStatus === 'offline'}
                className="btn-send"
                title="Send message"
                aria-label="Send message"
              >
                <Send size={14} />
                <span>Send</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
