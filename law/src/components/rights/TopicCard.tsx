import { Link } from '@/i18n/navigation';
import { Card } from '@/components/ui/Card';
import { KnowledgeEntry } from '@/types';

export function TopicCard({ entry }: { entry: KnowledgeEntry }) {
  return (
    <Link href={`/rights/${entry.categoryId}/${entry.id}`}>
      <Card className="group cursor-pointer transition-all hover:-translate-y-0.5 hover:shadow-md">
        <h3 className="mb-2 font-semibold text-slate-900 group-hover:text-primary-600 dark:text-white dark:group-hover:text-primary-400">
          {entry.question}
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
          {entry.answer.simple}
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {entry.keywords.slice(0, 4).map((keyword) => (
            <span
              key={keyword}
              className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-700 dark:text-slate-400"
            >
              {keyword}
            </span>
          ))}
        </div>
      </Card>
    </Link>
  );
}
