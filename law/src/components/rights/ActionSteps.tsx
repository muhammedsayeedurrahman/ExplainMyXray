'use client';

import { useTranslations } from 'next-intl';
import { ActionStep } from '@/types';
import { StaggerContainer, StaggerItem } from '@/components/ui/animations';

export function ActionSteps({ steps }: { steps: ActionStep[] }) {
  const t = useTranslations('rights');

  return (
    <div className="mt-8">
      <h3 className="mb-4 text-lg font-semibold text-slate-900 dark:text-white">
        {t('actionSteps')}
      </h3>
      <StaggerContainer className="space-y-4" staggerDelay={0.15}>
        {steps.map((step) => (
          <StaggerItem key={step.step}>
            <div className="flex gap-4">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-100 text-sm font-bold text-primary-700 dark:bg-primary-900/30 dark:text-primary-400">
                {step.step}
              </div>
              <div>
                <h4 className="font-medium text-slate-900 dark:text-white">
                  {step.title}
                </h4>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                  {step.description}
                </p>
              </div>
            </div>
          </StaggerItem>
        ))}
      </StaggerContainer>
    </div>
  );
}
