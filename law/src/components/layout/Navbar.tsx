'use client';

import { useState, useEffect } from 'react';
import { useTranslations } from 'next-intl';
import { Link, usePathname } from '@/i18n/navigation';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ThemeToggle } from './ThemeToggle';
import { MobileMenu } from './MobileMenu';
import { cn } from '@/lib/utils';

export function Navbar() {
  const t = useTranslations('nav');
  const common = useTranslations('common');
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    function onScroll() {
      setScrolled(window.scrollY > 10);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const links = [
    { href: '/', label: t('home') },
    { href: '/rights', label: t('rights') },
    { href: '/emergency', label: t('emergency') },
    { href: '/chat', label: t('chat') },
    { href: '/about', label: t('about') },
    { href: '/contact', label: t('contact') },
  ];

  return (
    <header
      className={cn(
        'sticky top-0 z-40 bg-white/80 backdrop-blur-md dark:bg-slate-900/80 transition-all duration-300',
        scrolled
          ? 'border-b shadow-sm dark:border-slate-700'
          : 'border-b border-transparent'
      )}
    >
      <div className="container-custom flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl" role="img" aria-label="scales of justice">
            &#9878;
          </span>
          <span className="text-lg font-bold text-slate-900 dark:text-white">
            {common('siteName')}
          </span>
        </Link>

        <nav className="hidden md:block" aria-label="Main navigation">
          <ul className="flex items-center gap-1">
            {links.map((link) => {
              const isActive = link.href === '/'
                ? pathname === '/'
                : pathname.startsWith(link.href);
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className={cn(
                      'rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary-50 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400'
                        : 'text-slate-700 hover:bg-slate-100 hover:text-primary-600 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-primary-400'
                    )}
                  >
                    {link.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="flex items-center gap-1">
          <LanguageSwitcher />
          <ThemeToggle />
          <MobileMenu />
        </div>
      </div>
    </header>
  );
}
