'use client';

import { motion, useReducedMotion } from 'framer-motion';
import { ChatMessage as ChatMessageType } from '@/types';
import { formatDate } from '@/lib/utils';
import { ChatResponse } from './ChatResponse';

export function ChatMessageBubble({ message }: { message: ChatMessageType }) {
  const isUser = message.type === 'user';
  const reduced = useReducedMotion();

  return (
    <motion.div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
      initial={reduced ? undefined : { opacity: 0, x: isUser ? 20 : -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100'
        }`}
      >
        {isUser ? (
          <p className="text-sm">{message.content}</p>
        ) : message.results && message.results.length > 0 ? (
          <ChatResponse results={message.results} />
        ) : (
          <p className="text-sm">{message.content}</p>
        )}
        <p
          className={`mt-1 text-xs ${
            isUser ? 'text-primary-200' : 'text-slate-400 dark:text-slate-500'
          }`}
        >
          {formatDate(message.timestamp)}
        </p>
      </div>
    </motion.div>
  );
}
