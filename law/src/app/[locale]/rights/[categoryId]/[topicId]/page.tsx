import { getLocale, getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { Link } from '@/i18n/navigation';
import { categories } from '@/data/categories';
import { loadEntry, getRelatedEntries } from '@/lib/knowledge-loader';
import { Breadcrumb } from '@/components/rights/Breadcrumb';
import { ThreeLayerExplanation } from '@/components/rights/ThreeLayerExplanation';
import { ActionSteps } from '@/components/rights/ActionSteps';
import { Card } from '@/components/ui/Card';

interface Props {
  params: { categoryId: string; topicId: string };
}

export default async function TopicPage({ params: { categoryId, topicId } }: Props) {
  const locale = await getLocale();
  const t = await getTranslations('rights');
  const ct = await getTranslations('categories');

  const category = categories.find((c) => c.id === categoryId);
  if (!category) notFound();

  const entry = await loadEntry(locale, topicId);
  if (!entry) notFound();

  const related = entry.relatedIds.length > 0
    ? await getRelatedEntries(locale, entry.relatedIds)
    : [];

  return (
    <div className="py-8 md:py-12">
      <div className="container-custom">
        <Breadcrumb
          items={[
            { label: t('title'), href: '/rights' },
            { label: ct(`${categoryId}.title`), href: `/rights/${categoryId}` },
            { label: entry.question },
          ]}
        />

        <div className="mx-auto max-w-4xl">
          <h1 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white md:text-3xl">
            {entry.question}
          </h1>

          <Card className="mb-8">
            <ThreeLayerExplanation answer={entry.answer} />
          </Card>

          {entry.actionSteps.length > 0 && (
            <Card className="mb-8">
              <ActionSteps steps={entry.actionSteps} />
            </Card>
          )}

          {related.length > 0 && (
            <div>
              <h2 className="mb-4 text-xl font-semibold text-slate-900 dark:text-white">
                {t('relatedTopics')}
              </h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {related.map((item) => (
                  <Link key={item.id} href={`/rights/${item.categoryId}/${item.id}`}>
                    <Card className="group cursor-pointer hover:shadow-md">
                      <h3 className="font-medium text-slate-900 group-hover:text-primary-600 dark:text-white dark:group-hover:text-primary-400">
                        {item.question}
                      </h3>
                      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400 line-clamp-2">
                        {item.answer.simple}
                      </p>
                    </Card>
                  </Link>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
