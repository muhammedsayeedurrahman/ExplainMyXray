'use client';

import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { Button } from '@/components/ui/Button';
import { PageTransition } from '@/components/ui/PageTransition';

export default function NotFoundPage() {
  const t = useTranslations('notFound');

  return (
    <PageTransition>
      <div className="flex min-h-[60vh] items-center justify-center py-16">
        <div className="text-center">
          <p className="text-6xl font-bold text-primary-600 dark:text-primary-400">404</p>
          <h1 className="mt-4 text-2xl font-bold text-slate-900 dark:text-white">
            {t('title')}
          </h1>
          <p className="mt-2 text-slate-600 dark:text-slate-400">
            {t('message')}
          </p>
          <div className="mt-6">
            <Link href="/">
              <Button>{t('goHome')}</Button>
            </Link>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
