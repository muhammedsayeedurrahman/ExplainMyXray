'use client';

import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { Button } from '@/components/ui/Button';
import { motion, useReducedMotion } from 'framer-motion';

export function HeroSection() {
  const t = useTranslations('home');
  const reduced = useReducedMotion();

  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-primary-600 via-primary-700 to-primary-900 py-20 text-white md:py-28">
      {/* Animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary-600/0 via-accent-500/10 to-primary-600/0 animate-gradient-shift" style={{ backgroundSize: '200% 100%' }} />

      {/* Pattern overlay */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDM0djItSDI0di0yaDEyek0zNiAyNHYySDI0di0yaDEyeiIvPjwvZz48L2c+PC9zdmc+')] opacity-30" />

      {/* Floating decorative circles */}
      <div className="absolute left-10 top-20 h-64 w-64 rounded-full bg-white/5 animate-float" aria-hidden="true" />
      <div className="absolute right-10 bottom-10 h-48 w-48 rounded-full bg-accent-500/10 animate-float-delayed" aria-hidden="true" />
      <div className="absolute left-1/2 top-10 h-32 w-32 rounded-full bg-white/5 animate-float-slow" aria-hidden="true" />

      <div className="container-custom relative">
        <div className="mx-auto max-w-3xl text-center">
          <motion.h1
            className="text-4xl font-bold tracking-tight md:text-5xl lg:text-6xl"
            initial={reduced ? undefined : { opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-gradient from-white via-primary-100 to-white">{t('heroTitle')}</span>
          </motion.h1>
          <motion.p
            className="mt-6 text-lg text-primary-100 md:text-xl"
            initial={reduced ? undefined : { opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
          >
            {t('heroSubtitle')}
          </motion.p>
          <motion.div
            className="mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center"
            initial={reduced ? undefined : { opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <Link href="/rights">
              <Button size="lg" className="relative overflow-hidden bg-white text-primary-700 hover:bg-primary-50">
                <span className="relative z-10">{t('heroCta')}</span>
                <span className="absolute inset-0 shimmer-effect" aria-hidden="true" />
              </Button>
            </Link>
            <Link href="/emergency">
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10">
                {t('heroEmergency')}
              </Button>
            </Link>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
