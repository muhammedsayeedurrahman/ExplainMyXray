'use client';

import { useState, FormEvent } from 'react';
import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Button } from '@/components/ui/Button';
import { PageTransition } from '@/components/ui/PageTransition';

export default function ContactPage() {
  const t = useTranslations('contact');
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setSubmitted(true);
  }

  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <div className="mx-auto max-w-2xl">
            <h1 className="mb-2 text-3xl font-bold text-slate-900 dark:text-white md:text-4xl">
              {t('title')}
            </h1>
            <p className="mb-8 text-lg text-slate-600 dark:text-slate-400">
              {t('subtitle')}
            </p>

            <Card>
              {submitted ? (
                <div className="py-8 text-center">
                  <div className="mb-4 text-4xl">&#10003;</div>
                  <p className="text-lg font-medium text-green-600 dark:text-green-400">
                    {t('success')}
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-5">
                  <div>
                    <label
                      htmlFor="name"
                      className="mb-1.5 block text-sm font-medium text-slate-700 dark:text-slate-300"
                    >
                      {t('nameLabel')}
                    </label>
                    <Input
                      id="name"
                      name="name"
                      required
                      placeholder={t('namePlaceholder')}
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="email"
                      className="mb-1.5 block text-sm font-medium text-slate-700 dark:text-slate-300"
                    >
                      {t('emailLabel')}
                    </label>
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      required
                      placeholder={t('emailPlaceholder')}
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="message"
                      className="mb-1.5 block text-sm font-medium text-slate-700 dark:text-slate-300"
                    >
                      {t('messageLabel')}
                    </label>
                    <Textarea
                      id="message"
                      name="message"
                      required
                      placeholder={t('messagePlaceholder')}
                    />
                  </div>

                  <Button type="submit" size="lg" className="w-full">
                    {t('submit')}
                  </Button>
                </form>
              )}
            </Card>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
