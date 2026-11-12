import React, { useState } from 'react';
import { Bot, Check, CheckCircle2, Code2, Copy, Cpu, Sparkles, User, AlertTriangle } from 'lucide-react';
import { ChatMessage, RoutingCategory } from '../types';
import { CodeBlock } from './CodeBlock';

interface MessageItemProps {
  message: ChatMessage;
  onRegenerate?: () => void;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRegenerate }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  const renderCategoryBadge = (category?: RoutingCategory, model?: string) => {
    if (!category || category === 'UNKNOWN') return null;

    let bg = 'badge-coding';
    let icon = <Code2 size={13} />;
    let label = 'CODING';

    if (category === 'UNIT_TESTING') {
      bg = 'badge-testing';
      icon = <CheckCircle2 size={13} />;
      label = 'UNIT TESTING';
    } else if (category === 'REASONING') {
      bg = 'badge-reasoning';
      icon = <Cpu size={13} />;
      label = 'REASONING';
    }

    return (
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`routing-badge ${bg}`}>
          {icon}
          <span>{label}</span>
        </span>
        {model && (
          <span className="model-target-pill" title={`Forwarded to ${model}`}>
            <Sparkles size={11} />
            <span>{model}</span>
          </span>
        )}
      </div>
    );
  };

  // Safe markdown text block parser
  const renderFormattedContent = (rawText: string) => {
    if (!rawText) return null;

    // Split text by triple backtick code blocks
    const parts = rawText.split(/(```[\s\S]*?```)/g);

    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const firstLineEnd = part.indexOf('\n');
        let lang = '';
        let code = '';
        if (firstLineEnd !== -1) {
          lang = part.slice(3, firstLineEnd).trim();
          code = part.slice(firstLineEnd + 1, -3);
        } else {
          code = part.slice(3, -3);
        }
        return <CodeBlock key={index} language={lang} code={code} />;
      }

      // Format non-code lines
      const lines = part.split('\n');
      return (
        <div key={index} className="prose-segment">
          {lines.map((line, lIdx) => {
            if (!line.trim()) {
              return <div key={lIdx} className="h-3" />;
            }

            // Headers
            if (line.startsWith('### ')) {
              return <h3 key={lIdx} className="text-base font-semibold text-zinc-100 my-2">{line.slice(4)}</h3>;
            }
            if (line.startsWith('## ')) {
              return <h2 key={lIdx} className="text-lg font-bold text-zinc-100 my-2">{line.slice(3)}</h2>;
            }
            if (line.startsWith('# ')) {
              return <h1 key={lIdx} className="text-xl font-bold text-zinc-100 my-3">{line.slice(2)}</h1>;
            }

            // List items
            if (line.startsWith('- ') || line.startsWith('* ')) {
              return (
                <div key={lIdx} className="flex gap-2 ml-3 my-1">
                  <span className="text-zinc-400">•</span>
                  <span>{parseInlineFormatting(line.slice(2))}</span>
                </div>
              );
            }

            return (
              <p key={lIdx} className="my-1 leading-relaxed">
                {parseInlineFormatting(line)}
              </p>
            );
          })}
        </div>
      );
    });
  };

  // Parse inline bold, inline code, etc.
  const parseInlineFormatting = (text: string) => {
    // Split by inline code first: `code`
    const codeParts = text.split(/(`[^`]+`)/g);

    return codeParts.map((sub, i) => {
      if (sub.startsWith('`') && sub.endsWith('`') && sub.length > 2) {
        return (
          <code key={i} className="inline-code">
            {sub.slice(1, -1)}
          </code>
        );
      }

      // Split by bold: **bold**
      const boldParts = sub.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bSub, bI) => {
        if (bSub.startsWith('**') && bSub.endsWith('**') && bSub.length > 4) {
          return <strong key={bI} className="font-semibold text-zinc-200">{bSub.slice(2, -2)}</strong>;
        }
        return bSub;
      });
    });
  };

  return (
    <div className={`message-row ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-avatar">
        {isUser ? <User size={17} /> : <Bot size={17} />}
      </div>

      <div className="message-content-wrapper">
        <div className="message-header">
          <span className="message-sender">{isUser ? 'You' : 'ModelSync Router'}</span>
          <span className="message-timestamp">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>

          {!isUser && renderCategoryBadge(message.routedCategory, message.routedModel)}

          {message.durationMs !== undefined && (
            <span className="duration-pill" title="Round-trip duration">
              {(message.durationMs / 1000).toFixed(2)}s
            </span>
          )}
        </div>

        <div className="message-body">
          {renderFormattedContent(message.content)}

          {message.isStreaming && (
            <span className="streaming-cursor" title="Streaming response..." />
          )}

          {message.error && (
            <div className="error-banner">
              <AlertTriangle size={15} />
              <span>{message.error}</span>
            </div>
          )}
        </div>

        {/* Message Actions */}
        <div className="message-actions">
          <button
            type="button"
            onClick={handleCopy}
            className="action-btn"
            title="Copy message content"
          >
            {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          {!isUser && onRegenerate && (
            <button
              type="button"
              onClick={onRegenerate}
              className="action-btn"
              title="Regenerate this response"
            >
              <Sparkles size={13} />
              <span>Regenerate</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
