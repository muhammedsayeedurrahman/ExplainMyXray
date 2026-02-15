'use client';

import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { Card } from '@/components/ui/Card';
import { categories } from '@/data/categories';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

export function QuickAccessCards() {
  const t = useTranslations('home');
  const ct = useTranslations('categories');

  return (
    <section className="bg-slate-50 py-16 dark:bg-slate-800/50 md:py-24">
      <div className="container-custom">
        <SlideUp className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white md:text-4xl">
            {t('quickAccessTitle')}
          </h2>
          <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
            {t('quickAccessSubtitle')}
          </p>
        </SlideUp>

        <StaggerContainer className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4" staggerDelay={0.08}>
          {categories.map((category) => (
            <StaggerItem key={category.id}>
              <Link href={`/rights/${category.id}`}>
                <Card className="group cursor-pointer">
                  <div className={`mb-3 inline-flex rounded-lg p-3 text-2xl ${category.color}`}>
                    {category.icon}
                  </div>
                  <h3 className="mb-1 font-semibold text-slate-900 group-hover:text-primary-600 dark:text-white dark:group-hover:text-primary-400">
                    {ct(`${category.id}.title`)}
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    {ct(`${category.id}.description`)}
                  </p>
                </Card>
              </Link>
            </StaggerItem>
          ))}
        </StaggerContainer>
      </div>
    </section>
  );
}
