'use client';

import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { ThreeLayerAnswer } from '@/types';
import { cn } from '@/lib/utils';

const tabs = ['simpleExplanation', 'practicalGuide', 'legalBasis'] as const;
const tabKeys: (keyof ThreeLayerAnswer)[] = ['simple', 'practical', 'legal'];

export function ThreeLayerExplanation({ answer }: { answer: ThreeLayerAnswer }) {
  const [activeTab, setActiveTab] = useState(0);
  const t = useTranslations('rights');
  const reduced = useReducedMotion();

  return (
    <div>
      <div className="relative flex border-b dark:border-slate-700" role="tablist">
        {tabs.map((tab, index) => (
          <button
            key={tab}
            role="tab"
            aria-selected={activeTab === index}
            onClick={() => setActiveTab(index)}
            className={cn(
              'relative px-4 py-3 text-sm font-medium transition-colors',
              activeTab === index
                ? 'text-primary-600 dark:text-primary-400'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
            )}
          >
            {t(tab)}
            {activeTab === index && (
              <motion.div
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 dark:bg-primary-400"
                layoutId="tab-indicator"
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              />
            )}
          </button>
        ))}
      </div>

      <div className="mt-6" role="tabpanel">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={reduced ? undefined : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduced ? undefined : { opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="prose prose-slate max-w-none dark:prose-invert"
          >
            <p className="whitespace-pre-line text-slate-700 dark:text-slate-300">
              {answer[tabKeys[activeTab]]}
            </p>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
