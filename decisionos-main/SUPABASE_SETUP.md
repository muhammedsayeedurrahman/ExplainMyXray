# Supabase Setup Guide

This guide walks you through setting up Supabase for DecisionOS.

## Prerequisites

- Node.js 20+ installed
- Git installed
- A Supabase account (free tier is fine)

---

## Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up/login
2. Click **"New Project"**
3. Fill in project details:
   - **Name**: DecisionOS (or your preferred name)
   - **Database Password**: Generate a strong password (save it somewhere safe)
   - **Region**: Choose closest to your users (e.g., Mumbai for India market)
   - **Pricing Plan**: Free (upgradeable later)
4. Click **"Create new project"**
5. Wait ~2 minutes for provisioning

---

## Step 2: Get API Credentials

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy the following values:

   **Project URL**:
   ```
   https://xxxxxxxxxxx.supabase.co
   ```

   **Anon/Public Key** (under "Project API keys"):
   ```
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

3. Keep these values handy for the next step

---

## Step 3: Configure Environment Variables

1. In the DecisionOS project root, create a file named `.env.local`:

   ```bash
   cp .env.local.example .env.local
   ```

2. Open `.env.local` and replace the placeholder values:

   ```env
   NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

3. **DO NOT** commit `.env.local` to Git (it's already in `.gitignore`)

---

## Step 4: Run Database Migrations

Once you've designed the schema (Task #62), run migrations:

```bash
# Install Supabase CLI
npm install -g supabase

# Link to your Supabase project
supabase link --project-ref your-project-id

# Run migrations
supabase db push
```

---

## Step 5: Test Connection

Start the Next.js dev server:

```bash
npm run dev
```

Open your browser console and verify no Supabase connection errors.

---

## Step 6: Enable Realtime (Optional - Sprint 1 Week 3)

1. In Supabase dashboard, go to **Database** → **Replication**
2. Enable replication for these tables:
   - `tasks`
   - `handoffs`
   - `voice_recordings`

This allows real-time subscriptions for live updates.

---

## Step 7: Configure Authentication (Task #63)

1. In Supabase dashboard, go to **Authentication** → **Providers**
2. Enable **Email** provider (enabled by default)
3. (Optional) Enable **Google OAuth**:
   - Click **Google** provider
   - Add your Google OAuth credentials (Client ID, Client Secret)
   - Save

---

## Troubleshooting

### Error: "Invalid API key"
- Double-check `.env.local` has the correct `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Make sure you copied the **Anon key**, not the Service Role key

### Error: "Failed to fetch"
- Check if Supabase project is running (green status in dashboard)
- Verify `NEXT_PUBLIC_SUPABASE_URL` has the correct project URL
- Check your internet connection

### Error: "Row Level Security policy violation"
- You haven't set up RLS policies yet (Task #62 will handle this)
- For now, you can disable RLS on tables for testing (not recommended for production)

---

## Next Steps

After completing this setup:
- ✅ Move to **Task #62**: Design and implement database schema
- ✅ Create tables, RLS policies, and indexes
- ✅ Test CRUD operations with Supabase client

---

## Useful Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Next.js + Supabase Guide](https://supabase.com/docs/guides/getting-started/quickstarts/nextjs)
- [Supabase Auth with Next.js](https://supabase.com/docs/guides/auth/server-side/nextjs)
- [Row Level Security (RLS) Guide](https://supabase.com/docs/guides/auth/row-level-security)
