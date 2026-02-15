'use client';

import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { KnowledgeEntry } from '@/types';
import { cn } from '@/lib/utils';
import { Link } from '@/i18n/navigation';

export function ChatResponse({ results }: { results: KnowledgeEntry[] }) {
  const [activeTab, setActiveTab] = useState(0);
  const t = useTranslations('rights');
  const chatT = useTranslations('chat');

  if (results.length === 0) {
    return <p className="text-sm">{chatT('noMatch')}</p>;
  }

  const primary = results[0];
  const tabs = [
    { key: 'simpleExplanation', content: primary.answer.simple },
    { key: 'practicalGuide', content: primary.answer.practical },
    { key: 'legalBasis', content: primary.answer.legal },
  ] as const;

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium">{primary.question}</p>

      {/* Mini tab bar */}
      <div className="flex gap-1 rounded-lg bg-slate-200/50 p-0.5 dark:bg-slate-700/50">
        {tabs.map((tab, index) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(index)}
            className={cn(
              'flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors',
              activeTab === index
                ? 'bg-white text-primary-700 shadow-sm dark:bg-slate-600 dark:text-primary-300'
                : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
            )}
          >
            {t(tab.key)}
          </button>
        ))}
      </div>

      <p className="whitespace-pre-line text-sm">{tabs[activeTab].content}</p>

      {/* Action steps for primary result */}
      {primary.actionSteps.length > 0 && activeTab === 1 && (
        <div className="mt-2 space-y-1.5">
          {primary.actionSteps.map((step) => (
            <div key={step.step} className="flex gap-2 text-xs">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary-100 font-bold text-primary-700 dark:bg-primary-900/30 dark:text-primary-400">
                {step.step}
              </span>
              <span>{step.title}</span>
            </div>
          ))}
        </div>
      )}

      {/* Related results */}
      {results.length > 1 && (
        <div className="border-t pt-2 dark:border-slate-700">
          <p className="mb-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
            {chatT('relatedQuestions')}
          </p>
          <div className="space-y-1">
            {results.slice(1, 4).map((result) => (
              <Link
                key={result.id}
                href={`/rights/${result.categoryId}/${result.id}`}
                className="block text-xs text-primary-600 hover:underline dark:text-primary-400"
              >
                {result.question}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
