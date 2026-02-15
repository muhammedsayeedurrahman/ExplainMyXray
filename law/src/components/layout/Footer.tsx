import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';

export function Footer() {
  const t = useTranslations('nav');
  const common = useTranslations('common');

  return (
    <footer className="border-t bg-slate-50 dark:bg-slate-900/50 dark:border-slate-800">
      <div className="container-custom py-12">
        <div className="grid gap-8 md:grid-cols-3">
          <div>
            <Link href="/" className="flex items-center gap-2">
              <span className="text-2xl">&#9878;</span>
              <span className="text-lg font-bold text-slate-900 dark:text-white">
                {common('siteName')}
              </span>
            </Link>
            <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
              {common('tagline')}
            </p>
          </div>

          <div>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-white">
              {common('quickLinks')}
            </h3>
            <ul className="space-y-2">
              <li>
                <Link href="/rights" className="text-sm text-slate-600 hover:text-primary-600 dark:text-slate-400 dark:hover:text-primary-400">
                  {t('rights')}
                </Link>
              </li>
              <li>
                <Link href="/emergency" className="text-sm text-slate-600 hover:text-primary-600 dark:text-slate-400 dark:hover:text-primary-400">
                  {t('emergency')}
                </Link>
              </li>
              <li>
                <Link href="/chat" className="text-sm text-slate-600 hover:text-primary-600 dark:text-slate-400 dark:hover:text-primary-400">
                  {t('chat')}
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-900 dark:text-white">
              {common('important')}
            </h3>
            <ul className="space-y-2">
              <li>
                <Link href="/about" className="text-sm text-slate-600 hover:text-primary-600 dark:text-slate-400 dark:hover:text-primary-400">
                  {t('about')}
                </Link>
              </li>
              <li>
                <Link href="/contact" className="text-sm text-slate-600 hover:text-primary-600 dark:text-slate-400 dark:hover:text-primary-400">
                  {t('contact')}
                </Link>
              </li>
              <li>
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  {common('emergencyNumber')}
                </span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t pt-8 text-center dark:border-slate-800">
          <p className="text-sm text-slate-500 dark:text-slate-500">
            &copy; {new Date().getFullYear()} {common('siteName')}. {common('copyright')}
          </p>
        </div>
      </div>
    </footer>
  );
}
