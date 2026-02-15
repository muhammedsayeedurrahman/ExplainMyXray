import { getLocale, getTranslations } from 'next-intl/server';
import { loadKnowledgeBase } from '@/lib/knowledge-loader';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { PageTransition } from '@/components/ui/PageTransition';

export default async function ChatPage() {
  const locale = await getLocale();
  const t = await getTranslations('chat');
  const knowledgeBase = await loadKnowledgeBase(locale);

  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white md:text-4xl">
              {t('title')}
            </h1>
            <p className="mt-2 text-lg text-slate-600 dark:text-slate-400">
              {t('subtitle')}
            </p>
          </div>

          <div className="mx-auto max-w-3xl">
            <ChatInterface knowledgeBase={knowledgeBase} locale={locale} />
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
