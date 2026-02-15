'use client';

import { useEffect, useRef } from 'react';
import { useTranslations } from 'next-intl';
import { KnowledgeEntry } from '@/types';
import { useChat } from '@/hooks/useChat';
import { ChatMessageBubble } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { SuggestedQuestions } from './SuggestedQuestions';
import { FadeIn } from '@/components/ui/animations';

interface ChatInterfaceProps {
  knowledgeBase: KnowledgeEntry[];
  locale: string;
}

export function ChatInterface({ knowledgeBase, locale }: ChatInterfaceProps) {
  const { messages, isLoading, sendMessage } = useChat(knowledgeBase, locale);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const t = useTranslations('chat');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex h-[calc(100vh-280px)] min-h-[400px] max-h-[700px] flex-col rounded-xl border bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
      {/* Chat messages area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <FadeIn className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-4 text-4xl">&#9878;</div>
            <p className="mb-6 text-sm text-slate-600 dark:text-slate-400">
              {t('welcomeMessage')}
            </p>
            <SuggestedQuestions onSelect={sendMessage} />
          </FadeIn>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessageBubble key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-slate-100 px-4 py-3 dark:bg-slate-700">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:0.1s]" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:0.2s]" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="border-t px-4 py-2 dark:border-slate-700">
        <p className="text-xs text-slate-400 dark:text-slate-500">
          {t('disclaimer')}
        </p>
      </div>

      {/* Input area */}
      <div className="border-t p-4 dark:border-slate-700">
        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}
