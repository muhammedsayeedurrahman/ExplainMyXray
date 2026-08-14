# Deployment Guide

## Prerequisites

- [x] Supabase project created
- [x] OpenAI API key obtained
- [x] Vercel account created
- [x] Code pushed to GitHub

---

## 1. Deploy to Vercel

### Option A: Vercel Dashboard (Recommended)

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure project:
   - **Framework Preset:** Next.js
   - **Root Directory:** ./
   - **Build Command:** `npm run build`
   - **Output Directory:** `.next`

4. Add Environment Variables:

```
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_APP_URL=https://your-app.vercel.app
NODE_ENV=production
NEXT_PUBLIC_DEMO_MODE=false
```

5. Click **Deploy**

### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# Set environment variables
vercel env add NEXT_PUBLIC_SUPABASE_URL
vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY
vercel env add SUPABASE_SERVICE_ROLE_KEY
vercel env add OPENAI_API_KEY

# Deploy to production
vercel --prod
```

---

## 2. Configure Supabase for Production

### Update Redirect URLs

1. Go to Supabase Dashboard → Authentication → URL Configuration
2. Add production URL to **Redirect URLs:**
   ```
   https://your-app.vercel.app/**
   ```

### Update CORS

1. Go to Supabase Dashboard → API → CORS
2. Add your Vercel domain:
   ```
   https://your-app.vercel.app
   ```

### Verify RLS Policies

Run this query in Supabase SQL Editor to verify policies:

```sql
SELECT schemaname, tablename, policyname, permissive, roles, cmd
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
```

Should show policies for all tables:
- workspaces
- users
- tasks
- handoffs
- notifications
- meetings
- voice_recordings
- uploads

---

## 3. Test Production Deployment

### Smoke Tests

1. **Authentication:**
   - Sign up with new account
   - Verify email confirmation (if enabled)
   - Sign in
   - Sign out

2. **Database:**
   - Create a task
   - Update a task
   - Delete a task
   - Verify real-time updates (open two tabs)

3. **Voice:**
   - Record audio
   - Verify transcription
   - Check OpenAI API usage

4. **File Upload:**
   - Upload a document
   - Verify file appears in Supabase Storage
   - Download the file

### Performance Tests

```bash
# Lighthouse audit
npx lighthouse https://your-app.vercel.app --view

# Check bundle size
npm run build
# Review .next/analyze/
```

---

## 4. Set Up Monitoring

### Vercel Analytics

1. Go to Vercel Dashboard → Your Project → Analytics
2. Enable Web Analytics
3. Enable Speed Insights

### Supabase Monitoring

1. Go to Supabase Dashboard → Database → Query Performance
2. Monitor:
   - Active connections
   - Slow queries
   - Index usage

### OpenAI Usage

1. Go to https://platform.openai.com/usage
2. Set up usage alerts:
   - Monthly budget limit
   - Email notifications

---

## 5. Domain Setup (Optional)

### Add Custom Domain

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your domain: `app.yourdomain.com`
3. Follow DNS configuration steps

### Update Environment Variables

```
NEXT_PUBLIC_APP_URL=https://app.yourdomain.com
```

### Update Supabase Redirect URLs

Add custom domain to Supabase redirect URLs:
```
https://app.yourdomain.com/**
```

---

## 6. Post-Deployment Checklist

- [ ] All environment variables set correctly
- [ ] Supabase redirects configured
- [ ] Authentication flow works
- [ ] Database operations work
- [ ] Real-time subscriptions work
- [ ] Voice transcription works
- [ ] File uploads work
- [ ] Custom domain configured (if applicable)
- [ ] Analytics enabled
- [ ] Monitoring set up
- [ ] Backup strategy in place

---

## 7. Ongoing Maintenance

### Daily
- Monitor Vercel Analytics
- Check error logs in Vercel
- Review Supabase usage

### Weekly
- Review OpenAI API costs
- Check Supabase database size
- Review slow query logs

### Monthly
- Security updates: `npm audit`
- Dependency updates: `npm outdated`
- Review and optimize database indexes
- Backup database (Supabase auto-backups, but verify)

---

## Troubleshooting

### Build Fails

```bash
# Check build locally
npm run build

# Common issues:
# 1. TypeScript errors
# 2. Missing environment variables
# 3. Import path issues
```

### Authentication Fails

- Verify `NEXT_PUBLIC_SUPABASE_URL` is correct
- Check Supabase redirect URLs include production domain
- Verify anon key is correct

### Database Errors

- Check RLS policies are enabled
- Verify service role key is set
- Review Supabase logs

### Voice Transcription Fails

- Verify `OPENAI_API_KEY` is set
- Check OpenAI account has credits
- Review API usage limits

---

## Rollback Plan

If deployment has critical issues:

```bash
# Redeploy previous version
vercel rollback

# Or deploy specific commit
git checkout <previous-commit>
vercel --prod
```

---

## Support

- **Vercel Docs:** https://vercel.com/docs
- **Supabase Docs:** https://supabase.com/docs
- **Next.js Docs:** https://nextjs.org/docs
- **Project Issues:** [GitHub Issues]

---

Ready to deploy! 🚀
