'use client';

import { useTranslations } from 'next-intl';
import { categories } from '@/data/categories';
import { CategoryCard } from '@/components/rights/CategoryCard';
import { Breadcrumb } from '@/components/rights/Breadcrumb';
import { PageTransition } from '@/components/ui/PageTransition';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

export default function RightsPage() {
  const t = useTranslations('rights');

  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <Breadcrumb items={[{ label: t('title') }]} />

          <SlideUp className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white md:text-4xl">
              {t('title')}
            </h1>
            <p className="mt-2 text-lg text-slate-600 dark:text-slate-400">
              {t('subtitle')}
            </p>
          </SlideUp>

          <StaggerContainer className="grid gap-4 sm:grid-cols-2">
            {categories.map((category) => (
              <StaggerItem key={category.id}>
                <CategoryCard category={category} />
              </StaggerItem>
            ))}
          </StaggerContainer>
        </div>
      </div>
    </PageTransition>
  );
}
