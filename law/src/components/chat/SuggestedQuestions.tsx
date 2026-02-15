'use client';

import { useTranslations } from 'next-intl';
import { StaggerContainer, StaggerItem } from '@/components/ui/animations';

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void;
}

const questionKeys = [
  'suggested1', 'suggested2', 'suggested3', 'suggested4',
  'suggested5', 'suggested6', 'suggested7', 'suggested8',
] as const;

export function SuggestedQuestions({ onSelect }: SuggestedQuestionsProps) {
  const t = useTranslations('chat');

  return (
    <div className="mb-4">
      <p className="mb-3 text-sm font-medium text-slate-600 dark:text-slate-400">
        {t('suggestedQuestions')}
      </p>
      <StaggerContainer className="flex flex-wrap gap-2" staggerDelay={0.05}>
        {questionKeys.map((key) => (
          <StaggerItem key={key}>
            <button
              onClick={() => onSelect(t(key))}
              className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-700 transition-colors hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-primary-600 dark:hover:bg-primary-900/20 dark:hover:text-primary-400"
            >
              {t(key)}
            </button>
          </StaggerItem>
        ))}
      </StaggerContainer>
    </div>
  );
}
