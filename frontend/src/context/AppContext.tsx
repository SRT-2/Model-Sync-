import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import {
  AppConfig,
  ChatMessage,
  GenerationParams,
  HealthData,
  ModelCard,
  RoutingCategory,
  SessionHistoryItem,
} from '../types';
import { createChatCompletion, fetchHealth, fetchModels } from '../services/api';

export type ActiveTab = 'chat' | 'insights' | 'models' | 'history';
export type ConnectionStatus = 'online' | 'offline' | 'checking' | 'unauthorized';

interface AppContextType {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  config: AppConfig;
  updateConfig: (updates: Partial<AppConfig>) => void;
  health: HealthData | null;
  connectionStatus: ConnectionStatus;
  lastHealthCheckTime: number | null;
  availableModels: ModelCard[];
  messages: ChatMessage[];
  sessionHistory: SessionHistoryItem[];
  generationParams: GenerationParams;
  setGenerationParams: React.Dispatch<React.SetStateAction<GenerationParams>>;
  isGenerating: boolean;
  checkConnection: () => Promise<void>;
  sendMessage: (promptText: string) => Promise<void>;
  stopGeneration: () => void;
  regenerateLast: () => Promise<void>;
  clearChat: () => void;
  clearSessionHistory: () => void;
}

const AppContext = createContext<AppContextType | null>(null);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('chat');

  // Load stored Base URL or default to http://localhost:8000
  // Note: routerApiKey is NOT stored in localStorage by default to keep secrets safe in memory
  const [config, setConfig] = useState<AppConfig>(() => ({
    baseUrl: typeof window !== 'undefined' ? localStorage.getItem('modelsync_base_url') || 'http://localhost:8000' : 'http://localhost:8000',
    routerApiKey: '',
    selectedModel: 'smart-router',
  }));

  const [health, setHealth] = useState<HealthData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking');
  const [lastHealthCheckTime, setLastHealthCheckTime] = useState<number | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelCard[]>([]);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryItem[]>([]);
  const [generationParams, setGenerationParams] = useState<GenerationParams>({
    temperature: 0.7,
    top_p: 1.0,
    max_tokens: 2048,
    stream: true,
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const updateConfig = (updates: Partial<AppConfig>) => {
    setConfig((prev) => {
      const next = { ...prev, ...updates };
      if (updates.baseUrl) {
        localStorage.setItem('modelsync_base_url', updates.baseUrl);
      }
      return next;
    });
  };

  const checkConnection = async () => {
    setConnectionStatus('checking');
    try {
      const healthData = await fetchHealth(config);
      setHealth(healthData);
      setLastHealthCheckTime(Date.now());
      setConnectionStatus('online');

      // Also try fetching models list
      try {
        const models = await fetchModels(config);
        setAvailableModels(models);
        if (models.length > 0 && !models.some((m) => m.id === config.selectedModel)) {
          updateConfig({ selectedModel: models[0].id });
        }
      } catch (mErr: any) {
        if (mErr.status === 401) {
          setConnectionStatus('unauthorized');
        }
      }
    } catch (err: any) {
      if (err.status === 401) {
        setConnectionStatus('unauthorized');
      } else {
        setConnectionStatus('offline');
        setHealth(null);
      }
    }
  };

  useEffect(() => {
    checkConnection();
  }, [config.baseUrl, config.routerApiKey]);

  const sendMessage = async (promptText: string) => {
    if (!promptText.trim() || isGenerating) return;

    const userMsg: ChatMessage = {
      id: `usr-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
      role: 'user',
      content: promptText.trim(),
      timestamp: Date.now(),
    };

    const assistantMsgId = `asst-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
    const assistantPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: generationParams.stream,
    };

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setIsGenerating(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const historyForApi = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const result = await createChatCompletion(
        config,
        historyForApi,
        generationParams,
        {
          signal: controller.signal,
          onChunk: (chunk) => {
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id === assistantMsgId) {
                  return {
                    ...m,
                    content: m.content + (chunk.content || ''),
                    routedCategory: (chunk.routedCategory as RoutingCategory) || m.routedCategory,
                    routedModel: chunk.routedModel || m.routedModel,
                  };
                }
                return m;
              })
            );
          },
        }
      );

      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === assistantMsgId) {
            return {
              ...m,
              content: result.content,
              routedCategory: result.routedCategory,
              routedModel: result.routedModel,
              durationMs: result.durationMs,
              isStreaming: false,
              usage: result.usage,
            };
          }
          return m;
        })
      );

      // Add to session history
      const historyEntry: SessionHistoryItem = {
        id: assistantMsgId,
        timestamp: Date.now(),
        prompt: promptText.trim(),
        response: result.content,
        category: result.routedCategory,
        model: result.routedModel,
        durationMs: result.durationMs,
        tokens: result.usage?.total_tokens,
      };
      setSessionHistory((prev) => [historyEntry, ...prev]);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === assistantMsgId) {
              return {
                ...m,
                content: m.content || '(Generation canceled)',
                isStreaming: false,
              };
            }
            return m;
          })
        );
      } else {
        const errorMsg = err.detail || err.message || 'Unknown request failure';
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id === assistantMsgId) {
              return {
                ...m,
                content: m.content ? `${m.content}\n\n[Error: ${errorMsg}]` : `Error: ${errorMsg}`,
                error: errorMsg,
                isStreaming: false,
              };
            }
            return m;
          })
        );
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const regenerateLast = async () => {
    if (messages.length < 2 || isGenerating) return;
    // Find the last user message
    let lastUserPrompt = '';
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserPrompt = messages[i].content;
        break;
      }
    }
    if (!lastUserPrompt) return;

    // Pop the last assistant response
    setMessages((prev) => {
      const idx = prev.map((m) => m.role).lastIndexOf('assistant');
      if (idx !== -1) {
        return prev.slice(0, idx);
      }
      return prev;
    });

    await sendMessage(lastUserPrompt);
  };

  const clearChat = () => {
    stopGeneration();
    setMessages([]);
  };

  const clearSessionHistory = () => {
    setSessionHistory([]);
  };

  return (
    <AppContext.Provider
      value={{
        activeTab,
        setActiveTab,
        config,
        updateConfig,
        health,
        connectionStatus,
        lastHealthCheckTime,
        availableModels,
        messages,
        sessionHistory,
        generationParams,
        setGenerationParams,
        isGenerating,
        checkConnection,
        sendMessage,
        stopGeneration,
        regenerateLast,
        clearChat,
        clearSessionHistory,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
