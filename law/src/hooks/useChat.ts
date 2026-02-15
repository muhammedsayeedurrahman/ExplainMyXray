'use client';

import { useState, useCallback } from 'react';
import { ChatMessage, KnowledgeEntry, SearchResult } from '@/types';
import { searchKnowledgeBase } from '@/lib/chatbot';
import { generateId } from '@/lib/utils';

export function useChat(knowledgeBase: KnowledgeEntry[], locale: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(
    (query: string) => {
      if (!query.trim()) return;

      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        content: query,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      // Simulate brief thinking delay
      setTimeout(() => {
        const results: SearchResult[] = searchKnowledgeBase(
          query,
          knowledgeBase,
          locale
        );

        const botMessage: ChatMessage = {
          id: generateId(),
          type: 'bot',
          content:
            results.length > 0
              ? results[0].item.answer.simple
              : '',
          timestamp: new Date(),
          results: results.map((r) => r.item),
        };

        setMessages((prev) => [...prev, botMessage]);
        setIsLoading(false);
      }, 300);
    },
    [knowledgeBase, locale]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isLoading, sendMessage, clearMessages };
}
