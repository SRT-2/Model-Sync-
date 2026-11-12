import { AppConfig, GenerationParams, HealthData, ModelCard, RoutingCategory } from '../types';
import { ParsedSSEChunk, SSEStreamParser } from './sseParser';

export interface CompletionResult {
  content: string;
  routedCategory?: RoutingCategory;
  routedModel?: string;
  durationMs: number;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function getHeaders(config: AppConfig): HeadersInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (config.routerApiKey && config.routerApiKey.trim()) {
    headers['Authorization'] = `Bearer ${config.routerApiKey.trim()}`;
  }

  return headers;
}

export async function fetchHealth(config: AppConfig, signal?: AbortSignal): Promise<HealthData> {
  const url = `${config.baseUrl.replace(/\/+$/, '')}/health`;
  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: getHeaders(config),
      signal,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      throw new ApiError(`Health check failed with HTTP ${res.status}`, res.status, errText);
    }

    return await res.json();
  } catch (err: any) {
    if (err.name === 'AbortError') throw err;
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network connection failed', 0);
  }
}

export async function fetchModels(config: AppConfig, signal?: AbortSignal): Promise<ModelCard[]> {
  const url = `${config.baseUrl.replace(/\/+$/, '')}/v1/models`;
  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: getHeaders(config),
      signal,
    });

    if (!res.ok) {
      let detail = '';
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch {
        detail = await res.text().catch(() => '');
      }
      throw new ApiError(`Failed to fetch models (HTTP ${res.status})`, res.status, detail);
    }

    const data = await res.json();
    return Array.isArray(data?.data) ? data.data : [];
  } catch (err: any) {
    if (err.name === 'AbortError') throw err;
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network error fetching models', 0);
  }
}

export async function createChatCompletion(
  config: AppConfig,
  messages: { role: string; content: string }[],
  params: GenerationParams,
  options?: {
    signal?: AbortSignal;
    onChunk?: (chunk: ParsedSSEChunk) => void;
  }
): Promise<CompletionResult> {
  const startTime = Date.now();
  const url = `${config.baseUrl.replace(/\/+$/, '')}/v1/chat/completions`;

  const payload: Record<string, any> = {
    model: config.selectedModel || 'smart-router',
    messages,
    stream: params.stream,
  };

  if (params.temperature !== undefined) payload.temperature = params.temperature;
  if (params.top_p !== undefined) payload.top_p = params.top_p;
  if (params.max_tokens !== undefined) payload.max_tokens = params.max_tokens;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: getHeaders(config),
      body: JSON.stringify(payload),
      signal: options?.signal,
    });
  } catch (err: any) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(err.message || 'Failed to connect to router backend', 0);
  }

  // Header-based routing metadata
  let routedCategory = (response.headers.get('X-Routed-Category') || undefined) as RoutingCategory | undefined;
  let routedModel = response.headers.get('X-Routed-Model') || undefined;

  if (!response.ok) {
    let detail = '';
    try {
      const errJson = await response.json();
      detail = errJson.detail || (errJson.error && errJson.error.message) || JSON.stringify(errJson);
    } catch {
      detail = await response.text().catch(() => '');
    }

    if (response.status === 401) {
      throw new ApiError('Authentication required. Check your Router API Key in settings.', 401, detail);
    }
    if (response.status === 502) {
      throw new ApiError('Upstream provider error (e.g. invalid key or upstream downtime).', 502, detail);
    }
    if (response.status === 504) {
      throw new ApiError('Upstream model request timed out.', 504, detail);
    }
    throw new ApiError(`Request failed with HTTP ${response.status}`, response.status, detail);
  }

  // Handle Streaming
  if (params.stream) {
    if (!response.body) {
      throw new ApiError('Response body is missing for streaming', 500);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const parser = new SSEStreamParser();
    let accumulatedContent = '';

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const { events, isDone } = parser.feed(text);

        for (const event of events) {
          if (event.error) {
            throw new ApiError(`Stream error: ${event.error}`, 502, event.error);
          }

          if (event.routedCategory && !routedCategory) {
            routedCategory = event.routedCategory as RoutingCategory;
          }
          if (event.routedModel && !routedModel) {
            routedModel = event.routedModel;
          }

          if (event.content) {
            accumulatedContent += event.content;
          }

          if (options?.onChunk) {
            options.onChunk({
              ...event,
              routedCategory: routedCategory || event.routedCategory,
              routedModel: routedModel || event.routedModel,
            });
          }
        }

        if (isDone) break;
      }
    } finally {
      reader.releaseLock();
    }

    const durationMs = Date.now() - startTime;
    return {
      content: accumulatedContent,
      routedCategory,
      routedModel,
      durationMs,
    };
  }

  // Handle Non-streaming
  const data = await response.json();
  const durationMs = Date.now() - startTime;
  const choice = data.choices?.[0];
  const content = choice?.message?.content || '';

  return {
    content,
    routedCategory: (data.routed_category || routedCategory) as RoutingCategory | undefined,
    routedModel: data.routed_model || routedModel,
    durationMs,
    usage: data.usage,
  };
}
