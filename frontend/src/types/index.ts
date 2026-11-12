export type RoutingCategory = 'CODING' | 'UNIT_TESTING' | 'REASONING' | 'UNKNOWN';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  routedCategory?: RoutingCategory;
  routedModel?: string;
  durationMs?: number;
  isStreaming?: boolean;
  error?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ModelCard {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface HealthData {
  status: string;
  version: string;
  provider: string;
  configured_models: {
    coding: string;
    unit_testing: string;
    reasoning: string;
    classifier: string;
  };
  auth_enabled: boolean;
}

export interface GenerationParams {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream: boolean;
}

export interface AppConfig {
  baseUrl: string;
  routerApiKey: string;
  selectedModel: string;
}

export interface SessionHistoryItem {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  category?: RoutingCategory;
  model?: string;
  durationMs?: number;
  tokens?: number;
}
