import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

/**
 * Create a Supabase client for server-side usage
 *
 * This client is used in Server Components, Server Actions, and API Routes.
 * It reads and writes cookies to maintain session state.
 *
 * @example
 * ```tsx
 * import { createClient } from '@/lib/supabase/server';
 *
 * export default async function Page() {
 *   const supabase = await createClient();
 *   const { data } = await supabase.from('tasks').select('*');
 *   return <div>{JSON.stringify(data)}</div>;
 * }
 * ```
 */
export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set(name, value, options);
            });
          } catch (error) {
            // The `setAll` method was called from a Server Component.
            // This can be ignored if you have middleware refreshing
            // user sessions.
          }
        },
      },
    }
  );
}
