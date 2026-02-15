'use client';

import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { stateLegalServices } from '@/data/emergency-contacts';

export function StateSelector() {
  const [selectedState, setSelectedState] = useState('');
  const t = useTranslations('emergency');

  const service = stateLegalServices.find((s) => s.state === selectedState);

  return (
    <div>
      <h2 className="mb-4 text-2xl font-bold text-slate-900 dark:text-white">
        {t('stateLegalServices')}
      </h2>

      <select
        value={selectedState}
        onChange={(e) => setSelectedState(e.target.value)}
        className="mb-6 w-full max-w-sm rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm text-slate-900 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
        aria-label={t('selectState')}
      >
        <option value="">{t('selectState')}</option>
        {stateLegalServices.map((s) => (
          <option key={s.state} value={s.state}>
            {s.state}
          </option>
        ))}
      </select>

      {service && (
        <Card>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            {service.authority}
          </h3>
          <div className="mt-3 space-y-2 text-sm text-slate-600 dark:text-slate-400">
            <p className="flex items-center gap-2">
              <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
              </svg>
              <a href={`tel:${service.phone}`} className="font-medium text-primary-600 hover:underline dark:text-primary-400">
                {service.phone}
              </a>
            </p>
            <p className="flex items-start gap-2">
              <svg className="mt-0.5 h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              {service.address}
            </p>
            {service.website && (
              <p className="flex items-center gap-2">
                <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                </svg>
                <a href={service.website} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline dark:text-primary-400">
                  {service.website}
                </a>
              </p>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
