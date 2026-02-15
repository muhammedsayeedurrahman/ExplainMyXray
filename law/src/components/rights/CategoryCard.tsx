import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Category } from '@/types';

export function CategoryCard({ category }: { category: Category }) {
  const t = useTranslations('categories');
  const rights = useTranslations('rights');

  return (
    <Link href={`/rights/${category.id}`}>
      <Card className="group cursor-pointer transition-all hover:-translate-y-1 hover:shadow-lg">
        <div className="flex items-start gap-4">
          <div className={`shrink-0 rounded-lg p-3 text-2xl ${category.color}`}>
            {category.icon}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-slate-900 group-hover:text-primary-600 dark:text-white dark:group-hover:text-primary-400">
                {t(`${category.id}.title`)}
              </h3>
              <Badge variant="info">{category.topicCount} {rights('topics')}</Badge>
            </div>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              {t(`${category.id}.description`)}
            </p>
          </div>
        </div>
      </Card>
    </Link>
  );
}
