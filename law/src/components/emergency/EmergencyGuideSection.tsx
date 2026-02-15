import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { AccordionItem } from '@/components/ui/Accordion';
import { emergencyGuides } from '@/data/emergency-contacts';

export function EmergencyGuideSection() {
  const t = useTranslations('emergency');

  return (
    <div>
      <h2 className="mb-4 text-2xl font-bold text-slate-900 dark:text-white">
        {t('emergencyGuide')}
      </h2>

      <div className="space-y-4">
        {emergencyGuides.map((guide) => (
          <Card key={guide.id}>
            <AccordionItem title={guide.title} defaultOpen={false}>
              <p className="mb-4 text-sm text-slate-600 dark:text-slate-400">
                {guide.description}
              </p>
              <div className="space-y-3">
                {guide.steps.map((step, index) => (
                  <div key={index} className="flex gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary-100 text-xs font-bold text-primary-700 dark:bg-primary-900/30 dark:text-primary-400">
                      {index + 1}
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-slate-900 dark:text-white">
                        {step.title}
                      </h4>
                      <p className="mt-0.5 text-sm text-slate-600 dark:text-slate-400">
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </AccordionItem>
          </Card>
        ))}
      </div>
    </div>
  );
}
