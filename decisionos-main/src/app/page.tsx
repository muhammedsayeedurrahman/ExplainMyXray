'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

/**
 * Root page redirects to login
 * All authentication now handled via Supabase
 */
export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.push('/login');
  }, [router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-brand-red border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">Redirecting to login...</p>
      </div>
    </div>
  );
}
