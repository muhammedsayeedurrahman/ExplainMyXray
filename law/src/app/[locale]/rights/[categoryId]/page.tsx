import { getLocale, getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { categories } from '@/data/categories';
import { loadCategoryEntries } from '@/lib/knowledge-loader';
import { TopicCard } from '@/components/rights/TopicCard';
import { Breadcrumb } from '@/components/rights/Breadcrumb';
import { CategoryPageClient } from './CategoryPageClient';

interface Props {
  params: { categoryId: string };
}

export default async function CategoryPage({ params: { categoryId } }: Props) {
  const locale = await getLocale();
  const t = await getTranslations('rights');
  const ct = await getTranslations('categories');

  const category = categories.find((c) => c.id === categoryId);
  if (!category) notFound();

  const entries = await loadCategoryEntries(locale, categoryId);

  return (
    <CategoryPageClient
      categoryId={categoryId}
      categoryIcon={category.icon}
      categoryTitle={ct(`${categoryId}.title`)}
      categoryDescription={ct(`${categoryId}.description`)}
      rightsTitle={t('title')}
      entries={entries}
    />
  );
}
