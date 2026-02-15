'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-[60vh] items-center justify-center py-16">
      <div className="text-center">
        <p className="text-6xl font-bold text-red-500">Error</p>
        <h2 className="mt-4 text-2xl font-bold text-slate-900 dark:text-white">
          Something went wrong
        </h2>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          An unexpected error occurred. Please try again.
        </p>
        <div className="mt-6 flex flex-col items-center gap-3">
          <Button onClick={reset}>Try Again</Button>
          <Link
            href="/"
            className="text-sm font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
          >
            Go Home
          </Link>
        </div>
      </div>
    </div>
  );
}
