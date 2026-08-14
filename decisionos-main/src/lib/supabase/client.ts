import { createBrowserClient } from '@supabase/ssr';

/**
 * Create a Supabase client for browser-side usage
 *
 * This client is used in Client Components and browser-side code.
 * It automatically handles session management via cookies.
 *
 * @example
 * ```tsx
 * 'use client';
 * import { createClient } from '@/lib/supabase/client';
 *
 * const supabase = createClient();
 * const { data } = await supabase.from('tasks').select('*');
 * ```
 */
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}
