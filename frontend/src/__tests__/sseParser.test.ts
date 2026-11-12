import { describe, expect, it } from 'vitest';
import { SSEStreamParser } from '../services/sseParser';

describe('SSEStreamParser', () => {
  it('parses a single standard SSE event', () => {
    const parser = new SSEStreamParser();
    const chunk = 'data: {"id":"1","choices":[{"delta":{"content":"Hello world"}}]}\n\n';
    const { events, isDone } = parser.feed(chunk);

    expect(isDone).toBe(false);
    expect(events).toHaveLength(1);
    expect(events[0].content).toBe('Hello world');
  });

  it('parses multiple SSE events delivered in a single network read', () => {
    const parser = new SSEStreamParser();
    const chunk =
      'data: {"id":"1","choices":[{"delta":{"content":"Part 1"}}]}\n\n' +
      'data: {"id":"2","choices":[{"delta":{"content":"Part 2"}}]}\n\n';
    const { events, isDone } = parser.feed(chunk);

    expect(isDone).toBe(false);
    expect(events).toHaveLength(2);
    expect(events[0].content).toBe('Part 1');
    expect(events[1].content).toBe('Part 2');
  });

  it('correctly handles chunk fragmentation across multiple network reads', () => {
    const parser = new SSEStreamParser();

    // First network read delivers partial event
    const read1 = 'data: {"id":"1","choices":[{"del';
    const res1 = parser.feed(read1);
    expect(res1.events).toHaveLength(0);
    expect(res1.isDone).toBe(false);

    // Second network read completes the event
    const read2 = 'ta":{"content":"Fragmented"}}]}\n\n';
    const res2 = parser.feed(read2);
    expect(res2.events).toHaveLength(1);
    expect(res2.events[0].content).toBe('Fragmented');
  });

  it('detects [DONE] stream termination token', () => {
    const parser = new SSEStreamParser();
    const chunk = 'data: {"id":"1","choices":[{"delta":{"content":"End"}}]}\n\ndata: [DONE]\n\n';
    const { events, isDone } = parser.feed(chunk);

    expect(isDone).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].content).toBe('End');
  });

  it('extracts routing metadata from initial chunk', () => {
    const parser = new SSEStreamParser();
    const chunk =
      'data: {"id":"1","routed_category":"UNIT_TESTING","routed_model":"moonshotai/Kimi","choices":[{"delta":{}}]}\n\n';
    const { events } = parser.feed(chunk);

    expect(events).toHaveLength(1);
    expect(events[0].routedCategory).toBe('UNIT_TESTING');
    expect(events[0].routedModel).toBe('moonshotai/Kimi');
  });

  it('handles stream error messages from upstream gracefully', () => {
    const parser = new SSEStreamParser();
    const chunk = 'data: {"error":{"message":"Rate limit exceeded","type":"rate_limit"}}\n\n';
    const { events } = parser.feed(chunk);

    expect(events).toHaveLength(1);
    expect(events[0].error).toBe('Rate limit exceeded');
  });
});
