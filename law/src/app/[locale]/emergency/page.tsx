'use client';

import { useTranslations } from 'next-intl';
import { nationalHelplines } from '@/data/emergency-contacts';
import { HelplineCard } from '@/components/emergency/HelplineCard';
import { StateSelector } from '@/components/emergency/StateSelector';
import { EmergencyGuideSection } from '@/components/emergency/EmergencyGuideSection';
import { PageTransition } from '@/components/ui/PageTransition';
import { SlideUp, StaggerContainer, StaggerItem } from '@/components/ui/animations';

export default function EmergencyPage() {
  const t = useTranslations('emergency');

  return (
    <PageTransition>
      <div className="py-8 md:py-12">
        <div className="container-custom">
          <SlideUp className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white md:text-4xl">
              {t('title')}
            </h1>
            <p className="mt-2 text-lg text-slate-600 dark:text-slate-400">
              {t('subtitle')}
            </p>
          </SlideUp>

          {/* National Helplines */}
          <section className="mb-12">
            <SlideUp>
              <h2 className="mb-4 text-2xl font-bold text-slate-900 dark:text-white">
                {t('nationalHelplines')}
              </h2>
            </SlideUp>
            <StaggerContainer className="grid gap-4 md:grid-cols-2" staggerDelay={0.08}>
              {nationalHelplines.map((contact) => (
                <StaggerItem key={contact.number + contact.name}>
                  <HelplineCard contact={contact} />
                </StaggerItem>
              ))}
            </StaggerContainer>
          </section>

          {/* State Legal Services */}
          <section className="mb-12">
            <StateSelector />
          </section>

          {/* Emergency Guides */}
          <section>
            <EmergencyGuideSection />
          </section>
        </div>
      </div>
    </PageTransition>
  );
}
