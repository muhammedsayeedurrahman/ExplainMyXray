import { HTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium',
        {
          'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300':
            variant === 'default',
          'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400':
            variant === 'success',
          'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400':
            variant === 'warning',
          'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400':
            variant === 'danger',
          'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400':
            variant === 'info',
        },
        className
      )}
      {...props}
    />
  );
}
