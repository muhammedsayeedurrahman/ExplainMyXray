import { useTranslations } from 'next-intl';

export function SkipToContent() {
  const t = useTranslations('common');

  return (
    <a href="#main-content" className="skip-to-content">
      {t('skipToContent')}
    </a>
  );
}
