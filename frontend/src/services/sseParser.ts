export interface ParsedSSEChunk {
  content?: string;
  finishReason?: string | null;
  routedCategory?: string;
  routedModel?: string;
  error?: string;
  raw?: any;
}

export class SSEStreamParser {
  private buffer = '';

  /**
   * Feed a new text chunk received from a network read into the parser.
   * Yields any fully formed SSE events found.
   */
  public feed(chunk: string): { events: ParsedSSEChunk[]; isDone: boolean } {
    this.buffer += chunk;
    const events: ParsedSSEChunk[] = [];
    let isDone = false;

    // SSE events are separated by double newlines (\n\n or \r\n\r\n)
    const blocks = this.buffer.split(/\r?\n\r?\n/);

    // The last element is either empty (if buffer ended in \n\n) or an incomplete event
    this.buffer = blocks.pop() || '';

    for (const block of blocks) {
      if (!block.trim()) continue;

      const lines = block.split(/\r?\n/);
      let dataStr = '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const val = line.slice(5).trim();
          dataStr = dataStr ? `${dataStr}\n${val}` : val;
        }
      }

      if (!dataStr) continue;

      if (dataStr === '[DONE]') {
        isDone = true;
        break;
      }

      try {
        const parsed = JSON.parse(dataStr);

        if (parsed.error) {
          events.push({
            error: typeof parsed.error === 'string' ? parsed.error : parsed.error.message || 'Stream error',
            raw: parsed,
          });
          continue;
        }

        const choice = parsed.choices?.[0];
        const content = choice?.delta?.content || choice?.message?.content || choice?.text || '';
        const finishReason = choice?.finish_reason || null;

        events.push({
          content,
          finishReason,
          routedCategory: parsed.routed_category,
          routedModel: parsed.routed_model,
          raw: parsed,
        });
      } catch {
        // Skip unparseable chunks gracefully
      }
    }

    return { events, isDone };
  }

  public reset(): void {
    this.buffer = '';
  }
}
