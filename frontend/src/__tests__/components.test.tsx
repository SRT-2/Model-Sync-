import React from 'react';
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { CodeBlock } from '../components/CodeBlock';
import { MessageItem } from '../components/MessageItem';
import { ChatMessage } from '../types';

describe('UI Components', () => {
  it('renders CodeBlock with language header and copy button', () => {
    render(<CodeBlock language="python" code="def foo(): pass" />);

    expect(screen.getByText('python')).toBeInTheDocument();
    expect(screen.getByText('def foo(): pass')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy code/i })).toBeInTheDocument();
  });

  it('renders MessageItem with CODING routing badge and model tag', () => {
    const msg: ChatMessage = {
      id: 'msg-1',
      role: 'assistant',
      content: 'Here is your Python script:\n```python\nprint(1)\n```',
      timestamp: 1700000000000,
      routedCategory: 'CODING',
      routedModel: 'zai-org/GLM-5.1',
      durationMs: 1450,
    };

    render(<MessageItem message={msg} />);

    expect(screen.getByText('CODING')).toBeInTheDocument();
    expect(screen.getByText('zai-org/GLM-5.1')).toBeInTheDocument();
    expect(screen.getByText('1.45s')).toBeInTheDocument();
    expect(screen.getByText('print(1)')).toBeInTheDocument();
  });

  it('renders MessageItem with UNIT_TESTING routing badge', () => {
    const msg: ChatMessage = {
      id: 'msg-2',
      role: 'assistant',
      content: 'Here are the pytest tests',
      timestamp: 1700000000000,
      routedCategory: 'UNIT_TESTING',
      routedModel: 'moonshotai/Kimi-K2.6',
    };

    render(<MessageItem message={msg} />);

    expect(screen.getByText('UNIT TESTING')).toBeInTheDocument();
    expect(screen.getByText('moonshotai/Kimi-K2.6')).toBeInTheDocument();
  });
});
