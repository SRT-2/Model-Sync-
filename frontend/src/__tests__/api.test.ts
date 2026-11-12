import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, createChatCompletion, fetchHealth, fetchModels } from '../services/api';
import { AppConfig, GenerationParams } from '../types';

const mockConfig: AppConfig = {
  baseUrl: 'http://localhost:8000',
  routerApiKey: 'test-secret',
  selectedModel: 'smart-router',
};

describe('API Client Service', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetchHealth queries /health with Authorization header', async () => {
    const mockHealth = {
      status: 'ok',
      version: '2.0.0',
      provider: 'openrouter',
      configured_models: {
        coding: 'qwen/qwen3.8-27b:free',
        unit_testing: 'google/gemma-4-26b-a4b-it:free',
        reasoning: 'nvidia/nemotron-3-ultra-550b-a55b:free',
        classifier: 'poolside/laguna-s-2.1:free',
      },
      auth_enabled: true,
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await fetchHealth(mockConfig);

    expect(fetchMock).toHaveBeenCalledWith('http://localhost:8000/health', expect.objectContaining({
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer test-secret',
      },
    }));
    expect(result.status).toBe('ok');
    expect(result.provider).toBe('openrouter');
  });

  it('fetchModels throws ApiError on 401 unauthorized', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'Invalid API key' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchModels(mockConfig)).rejects.toThrow(ApiError);
  });

  it('createChatCompletion handles non-streaming request with routing metadata', async () => {
    const mockResponse = {
      id: 'chatcmpl-123',
      choices: [
        {
          message: { role: 'assistant', content: 'Here is the code' },
          finish_reason: 'stop',
        },
      ],
      routed_category: 'CODING',
      routed_model: 'qwen/qwen3.8-27b:free',
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({
        'X-Routed-Category': 'CODING',
        'X-Routed-Model': 'qwen/qwen3.8-27b:free',
      }),
      json: async () => mockResponse,
    });
    vi.stubGlobal('fetch', fetchMock);

    const params: GenerationParams = {
      temperature: 0.5,
      stream: false,
    };

    const result = await createChatCompletion(
      mockConfig,
      [{ role: 'user', content: 'Write a quicksort function' }],
      params
    );

    expect(result.content).toBe('Here is the code');
    expect(result.routedCategory).toBe('CODING');
    expect(result.routedModel).toBe('qwen/qwen3.8-27b:free');
    expect(result.usage?.total_tokens).toBe(30);
  });
});
