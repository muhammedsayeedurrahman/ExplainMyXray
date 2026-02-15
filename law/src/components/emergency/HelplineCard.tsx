import { useTranslations } from 'next-intl';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { EmergencyContact } from '@/types';

export function HelplineCard({ contact }: { contact: EmergencyContact }) {
  const t = useTranslations('emergency');

  return (
    <Card className="flex items-center justify-between">
      <div>
        <h3 className="font-semibold text-slate-900 dark:text-white">
          {contact.name}
        </h3>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          {contact.description}
        </p>
        <div className="mt-2 flex gap-2">
          {contact.available24x7 && (
            <Badge variant="success">{t('available24x7')}</Badge>
          )}
          {contact.tollFree && (
            <Badge variant="info">{t('tollFree')}</Badge>
          )}
        </div>
      </div>
      <a
        href={`tel:${contact.number}`}
        className="relative ml-4 shrink-0 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-700"
        aria-label={`Call ${contact.name} at ${contact.number}`}
      >
        {/* Pulse ring animation */}
        <span className="absolute inset-0 rounded-lg bg-red-600 animate-pulse-ring" aria-hidden="true" />
        <div className="relative text-center">
          <div className="text-lg font-bold">{contact.number}</div>
          <div className="text-xs">{t('callNow')}</div>
        </div>
      </a>
    </Card>
  );
}
