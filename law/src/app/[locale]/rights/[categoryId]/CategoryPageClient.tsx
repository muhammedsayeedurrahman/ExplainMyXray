'use client';

import { KnowledgeEntry } from '@/types';
import { TopicCard } from '@/components/rights/TopicCard';
import { Breadcrumb } from '@/components/rights/Breadcrumb';
import { PageTransition } from '@/components/ui/PageTransition';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

interface CategoryPageClientProps {
  categoryId: string;
  categoryIcon: string;
  categoryTitle: string;
  categoryDescription: string;
  rightsTitle: string;
  entries: KnowledgeEntry[];
}

export function CategoryPageClient({
  categoryId,
  categoryIcon,
  categoryTitle,
  categoryDescription,
  rightsTitle,
  entries,
}: CategoryPageClientProps) {
  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <Breadcrumb
            items={[
              { label: rightsTitle, href: '/rights' },
              { label: categoryTitle },
            ]}
          />

          <SlideUp className="mb-8">
            <div className="flex items-center gap-3">
              <span className="text-3xl">{categoryIcon}</span>
              <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
                {categoryTitle}
              </h1>
            </div>
            <p className="mt-2 text-lg text-slate-600 dark:text-slate-400">
              {categoryDescription}
            </p>
          </SlideUp>

          <StaggerContainer className="grid gap-4 sm:grid-cols-2">
            {entries.map((entry) => (
              <StaggerItem key={entry.id}>
                <TopicCard entry={entry} />
              </StaggerItem>
            ))}
          </StaggerContainer>

          {entries.length === 0 && (
            <p className="text-center text-slate-500 dark:text-slate-400 py-12">
              Content coming soon for this category.
            </p>
          )}
        </div>
      </div>
    </PageTransition>
  );
}
