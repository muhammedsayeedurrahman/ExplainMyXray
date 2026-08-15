import { createBrowserClient } from '@supabase/ssr';

/**
 * Create a Supabase client for client-side operations
 * This client includes authentication state management
 */
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}

/**
 * Singleton Supabase client for browser
 * Use this for client-side data fetching and mutations
 */
export const supabase = createClient();
