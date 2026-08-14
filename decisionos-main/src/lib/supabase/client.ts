import { createClient } from '@supabase/supabase-js';

/**
 * Database types (auto-generated from schema)
 * Run: supabase gen types typescript --project-id <project-id> > src/lib/supabase/types.ts
 */
export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      workspaces: {
        Row: {
          id: string;
          name: string;
          industry: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          industry?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          name?: string;
          industry?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      users: {
        Row: {
          id: string;
          workspace_id: string | null;
          email: string;
          full_name: string;
          role: 'owner' | 'sales' | 'production' | 'finance';
          avatar_url: string | null;
          phone: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          workspace_id?: string | null;
          email: string;
          full_name: string;
          role: 'owner' | 'sales' | 'production' | 'finance';
          avatar_url?: string | null;
          phone?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          workspace_id?: string | null;
          email?: string;
          full_name?: string;
          role?: 'owner' | 'sales' | 'production' | 'finance';
          avatar_url?: string | null;
          phone?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      tasks: {
        Row: {
          id: string;
          workspace_id: string;
          title: string;
          subtext: string | null;
          type: 'TASK' | 'REMINDER' | 'INVOICE' | 'APPROVAL';
          source: 'TEXT' | 'VOICE' | 'UPLOAD';
          category: 'CUSTOMER' | 'SUPPLIER' | 'INVOICE' | 'PAYMENT' | 'COMPLAINT' | 'OTHER';
          assigned_to: string | null;
          created_by: string;
          done: boolean;
          scheduled_date: string | null;
          scheduled_time: string | null;
          details_count: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          workspace_id: string;
          title: string;
          subtext?: string | null;
          type: 'TASK' | 'REMINDER' | 'INVOICE' | 'APPROVAL';
          source: 'TEXT' | 'VOICE' | 'UPLOAD';
          category: 'CUSTOMER' | 'SUPPLIER' | 'INVOICE' | 'PAYMENT' | 'COMPLAINT' | 'OTHER';
          assigned_to?: string | null;
          created_by: string;
          done?: boolean;
          scheduled_date?: string | null;
          scheduled_time?: string | null;
          details_count?: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          workspace_id?: string;
          title?: string;
          subtext?: string | null;
          type?: 'TASK' | 'REMINDER' | 'INVOICE' | 'APPROVAL';
          source?: 'TEXT' | 'VOICE' | 'UPLOAD';
          category?: 'CUSTOMER' | 'SUPPLIER' | 'INVOICE' | 'PAYMENT' | 'COMPLAINT' | 'OTHER';
          assigned_to?: string | null;
          created_by?: string;
          done?: boolean;
          scheduled_date?: string | null;
          scheduled_time?: string | null;
          details_count?: number;
          created_at?: string;
          updated_at?: string;
        };
      };
      handoffs: {
        Row: {
          id: string;
          workspace_id: string;
          from_user_id: string;
          to_user_id: string;
          title: string;
          description: string | null;
          instruction: string | null;
          status: 'pending' | 'submitted' | 'approved' | 'rejected';
          reply_text: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          workspace_id: string;
          from_user_id: string;
          to_user_id: string;
          title: string;
          description?: string | null;
          instruction?: string | null;
          status?: 'pending' | 'submitted' | 'approved' | 'rejected';
          reply_text?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          workspace_id?: string;
          from_user_id?: string;
          to_user_id?: string;
          title?: string;
          description?: string | null;
          instruction?: string | null;
          status?: 'pending' | 'submitted' | 'approved' | 'rejected';
          reply_text?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
    };
    Views: {};
    Functions: {};
    Enums: {};
  };
}

/**
 * Supabase client for Client Components
 * Safe to use in browser - uses anon key only
 * Uses placeholder values if not configured (demo mode - client won't be used)
 */
export const supabase = createClient<Database>(
  process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co',
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder-anon-key'
);

/**
 * Supabase client for Server Components and API Routes
 * Use this for server-side operations with elevated privileges
 */
export const createServerSupabaseClient = () => {
  if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.warn('Supabase environment variables not configured');
    return null;
  }

  return createClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    }
  );
};
