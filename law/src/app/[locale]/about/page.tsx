'use client';

import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { PageTransition } from '@/components/ui/PageTransition';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

export default function AboutPage() {
  const t = useTranslations('about');

  const sections = [
    { title: t('mission'), content: t('missionText') },
    { title: t('whatWeDo'), content: t('whatWeDoText') },
    { title: t('disclaimer'), content: t('disclaimerText') },
    { title: t('openSource'), content: t('openSourceText') },
  ];

  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <div className="mx-auto max-w-3xl">
            <SlideUp>
              <h1 className="mb-8 text-3xl font-bold text-slate-900 dark:text-white md:text-4xl">
                {t('title')}
              </h1>
            </SlideUp>

            <StaggerContainer className="space-y-6" staggerDelay={0.12}>
              {sections.map((section) => (
                <StaggerItem key={section.title}>
                  <Card>
                    <h2 className="mb-3 text-xl font-semibold text-slate-900 dark:text-white">
                      {section.title}
                    </h2>
                    <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                      {section.content}
                    </p>
                  </Card>
                </StaggerItem>
              ))}
            </StaggerContainer>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
