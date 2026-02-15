'use client';

import { useEffect, useRef, useState } from 'react';
import { useTranslations } from 'next-intl';
import { useReducedMotion } from 'framer-motion';
import { SlideUp } from '@/components/ui/animations';

function useCountUp(target: number, duration: number = 1500, inView: boolean) {
  const [count, setCount] = useState(0);
  const reduced = useReducedMotion();

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setCount(target);
      return;
    }

    let start = 0;
    const startTime = performance.now();

    function update(currentTime: number) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutCubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(eased * target);

      if (current !== start) {
        start = current;
        setCount(current);
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }, [target, duration, inView, reduced]);

  return count;
}

interface StatItemProps {
  value: number;
  suffix: string;
  label: string;
  inView: boolean;
}

function StatItem({ value, suffix, label, inView }: StatItemProps) {
  const count = useCountUp(value, 1500, inView);

  return (
    <div className="text-center">
      <div className="text-4xl font-bold text-gradient from-primary-600 to-accent-500 bg-clip-text text-transparent md:text-5xl">
        {count}{suffix}
      </div>
      <p className="mt-2 text-sm font-medium text-slate-600 dark:text-slate-400">
        {label}
      </p>
    </div>
  );
}

export function StatsSection() {
  const t = useTranslations('home');
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.3 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const stats = [
    { value: 40, suffix: '+', label: t('statTopics') },
    { value: 8, suffix: '', label: t('statCategories') },
    { value: 11, suffix: '', label: t('statLanguages') },
    { value: 24, suffix: '/7', label: t('statAvailability') },
  ];

  return (
    <section className="py-16 md:py-20">
      <div className="container-custom" ref={ref}>
        <SlideUp>
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4 md:gap-12">
            {stats.map((stat) => (
              <StatItem
                key={stat.label}
                value={stat.value}
                suffix={stat.suffix}
                label={stat.label}
                inView={inView}
              />
            ))}
          </div>
        </SlideUp>
      </div>
    </section>
  );
}
