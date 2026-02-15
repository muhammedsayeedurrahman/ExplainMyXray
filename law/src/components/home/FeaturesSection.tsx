'use client';

import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

const featureIcons = [
  // Simple Explanations
  <svg key="1" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>,
  // Smart Search
  <svg key="2" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>,
  // Emergency Contacts
  <svg key="3" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
  </svg>,
  // Multiple Languages
  <svg key="4" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
  </svg>,
];

export function FeaturesSection() {
  const t = useTranslations('home');

  const features = [
    { title: t('feature1Title'), desc: t('feature1Desc') },
    { title: t('feature2Title'), desc: t('feature2Desc') },
    { title: t('feature3Title'), desc: t('feature3Desc') },
    { title: t('feature4Title'), desc: t('feature4Desc') },
  ];

  return (
    <section className="relative py-16 md:py-24 overflow-hidden">
      {/* Decorative gradient blobs */}
      <div className="absolute -left-32 top-0 h-64 w-64 rounded-full bg-primary-100/50 blur-3xl dark:bg-primary-900/20" aria-hidden="true" />
      <div className="absolute -right-32 bottom-0 h-64 w-64 rounded-full bg-accent-100/50 blur-3xl dark:bg-accent-900/20" aria-hidden="true" />

      <div className="container-custom relative">
        <SlideUp className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white md:text-4xl">
            {t('featuresTitle')}
          </h2>
          <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
            {t('featuresSubtitle')}
          </p>
        </SlideUp>

        <StaggerContainer className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <StaggerItem key={index}>
              <Card className="text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400">
                  {featureIcons[index]}
                </div>
                <h3 className="mb-2 text-lg font-semibold text-slate-900 dark:text-white">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {feature.desc}
                </p>
              </Card>
            </StaggerItem>
          ))}
        </StaggerContainer>
      </div>
    </section>
  );
}
