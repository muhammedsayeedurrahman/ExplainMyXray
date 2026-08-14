'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Sun, Moon, ArrowLeft } from 'lucide-react';
import { signUp } from '@/lib/supabase/auth';
import { Role } from '@/config/roles';

export default function SignUpPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    fullName: '',
    role: 'owner' as Role,
    workspaceName: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [isFirstUser, setIsFirstUser] = useState(true);

  // Load theme preference on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
      setTheme('dark');
      document.documentElement.classList.add('dark');
    } else {
      setTheme('light');
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark');
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      setTheme('light');
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Validate password confirmation
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    // Validate password strength
    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters');
      setLoading(false);
      return;
    }

    // Validate workspace name for first user
    if (isFirstUser && !formData.workspaceName.trim()) {
      setError('Workspace name is required');
      setLoading(false);
      return;
    }

    try {
      await signUp({
        email: formData.email,
        password: formData.password,
        fullName: formData.fullName,
        role: formData.role,
        workspaceName: isFirstUser ? formData.workspaceName : undefined,
      });

      // Import helper to get dashboard path
      const { getDashboardPath } = await import('@/lib/supabase/helpers');
      const dashboardPath = getDashboardPath(formData.role);

      router.push(dashboardPath);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign up failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col md:flex-row bg-zinc-50 dark:bg-black text-zinc-900 dark:text-zinc-50 font-sans transition-colors duration-200">

      {/* Left Panel (Desktop only branding) */}
      <div className="hidden md:flex md:w-1/2 bg-zinc-950 dark:bg-[#1e1e20] text-white p-16 flex-col justify-between relative overflow-hidden border-r border-zinc-950 dark:border-zinc-800">
        {/* Subtle dot overlay */}
        <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.03)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none"></div>
        <div className="absolute inset-0 bg-gradient-to-b from-brand-red/5 to-transparent pointer-events-none"></div>

        {/* Logo */}
        <div className="flex items-center gap-3 z-10">
          <div className="w-9 h-9 bg-brand-red rounded-lg flex items-center justify-center font-logo font-black text-white text-xl shadow-[2px_2px_0px_0px_rgba(255,255,255,0.2)]">
            D
          </div>
          <span className="font-logo font-black text-2xl tracking-tight uppercase">
            Decision<span className="text-brand-red">OS</span>
          </span>
        </div>

        {/* Hero Section */}
        <div className="my-auto z-10 flex flex-col gap-6 max-w-lg">
          <div className="text-brand-red font-mono text-xs font-bold uppercase tracking-[0.2em] leading-none">
            CREATE YOUR OPERATIONAL BRAIN
          </div>
          <h1 className="text-5xl lg:text-6xl font-logo font-black uppercase tracking-tight leading-[1.05] text-white">
            BUILD YOUR<br />
            COMPANY<br />
            <span className="text-brand-red">BRAIN</span> IN<br />
            MINUTES.
          </h1>
          <p className="text-zinc-400 text-base font-normal leading-relaxed">
            Join thousands of SMEs using DecisionOS to streamline operations, delegate smarter, and grow faster.
          </p>
        </div>

        {/* Footer info */}
        <div className="flex items-center gap-2 z-10 text-zinc-500 dark:text-zinc-400 font-mono text-xs">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="text-brand-red animate-pulse" viewBox="0 0 256 256">
            <path d="M224,128a96,96,0,1,1-96-96A96,96,0,0,1,224,128Z"/>
          </svg>
          <span>Free 14-day trial &middot; No credit card required</span>
        </div>
      </div>

      {/* Right Panel (Sign-up form) */}
      <div className="w-full md:w-1/2 flex flex-col justify-center items-center p-8 md:p-16 relative min-h-screen bg-zinc-50 dark:bg-black">

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          aria-label="Toggle dark mode"
          className="fixed top-6 right-6 p-2.5 bg-white dark:bg-zinc-900 border border-zinc-950 dark:border-zinc-800 rounded-md hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors shadow-sm cursor-pointer z-50 text-zinc-900 dark:text-white"
        >
          {theme === 'light' ? (
            <Moon className="w-[18px] h-[18px]" />
          ) : (
            <Sun className="w-[18px] h-[18px]" />
          )}
        </button>

        {/* Back to Login */}
        <a
          href="/"
          className="fixed top-6 left-6 p-2.5 bg-white dark:bg-zinc-900 border border-zinc-950 dark:border-zinc-800 rounded-md hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors shadow-sm cursor-pointer z-50 text-zinc-900 dark:text-white flex items-center gap-2"
        >
          <ArrowLeft className="w-[18px] h-[18px]" />
          <span className="text-xs font-mono font-bold uppercase tracking-wider hidden sm:inline">Back</span>
        </a>

        {/* Sign-up Form Container */}
        <div className="w-full max-w-[500px] flex flex-col justify-center animate-fade-up">

          {/* Logo (Shown on mobile viewports only) */}
          <div className="flex items-center gap-2 mb-8 md:hidden">
            <div className="w-8 h-8 bg-brand-red rounded-lg flex items-center justify-center font-logo font-black text-white text-lg">
              D
            </div>
            <span className="font-logo font-black text-xl tracking-tight uppercase">
              Decision<span className="text-brand-red">OS</span>
            </span>
          </div>

          {/* Heading */}
          <h2 className="font-logo text-4xl font-black uppercase tracking-tight mb-1 text-zinc-900 dark:text-white">
            CREATE ACCOUNT
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-8 font-sans">
            Start your 14-day free trial.
          </p>

          {/* Alert messages */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-950/20 border border-brand-red text-xs text-brand-red font-semibold rounded-md">
              {error}
            </div>
          )}

          {/* Sign-up Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="fullName" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                Full Name
              </label>
              <input
                type="text"
                id="fullName"
                name="fullName"
                placeholder="Rajesh Sharma"
                required
                autoComplete="name"
                value={formData.fullName}
                onChange={handleChange}
                className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
              />
            </div>

            <div>
              <label htmlFor="email" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="rajesh@example.com"
                required
                autoComplete="email"
                value={formData.email}
                onChange={handleChange}
                className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
              />
            </div>

            <div>
              <label htmlFor="password" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                placeholder="At least 8 characters"
                required
                autoComplete="new-password"
                value={formData.password}
                onChange={handleChange}
                className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
              />
            </div>

            <div>
              <label htmlFor="confirmPassword" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                Confirm Password
              </label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                placeholder="Re-enter password"
                required
                autoComplete="new-password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
              />
            </div>

            <div>
              <label htmlFor="role" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                Your Role
              </label>
              <select
                id="role"
                name="role"
                required
                value={formData.role}
                onChange={handleChange}
                className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white rounded-md"
              >
                <option value="owner">Owner / CEO</option>
                <option value="sales">Sales Manager</option>
                <option value="production">Production Chief</option>
                <option value="finance">Finance Controller</option>
              </select>
            </div>

            {isFirstUser && (
              <div>
                <label htmlFor="workspaceName" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                  Workspace Name
                </label>
                <input
                  type="text"
                  id="workspaceName"
                  name="workspaceName"
                  placeholder="Sharma Textiles Pvt Ltd"
                  required
                  autoComplete="organization"
                  value={formData.workspaceName}
                  onChange={handleChange}
                  className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
                />
                <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Your company or organization name
                </p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-brand-red text-white font-bold uppercase tracking-wider py-3.5 border border-zinc-950 dark:border-zinc-800 hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] dark:hover:shadow-[3px_3px_0px_0px_rgba(255,59,48,0.4)] transition-all cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2 rounded-md"
            >
              {loading ? 'Creating Account...' : 'Create Account'}
            </button>
          </form>

          {/* Terms */}
          <p className="mt-6 text-xs text-zinc-500 dark:text-zinc-400 text-center">
            By signing up, you agree to our{' '}
            <a href="#" className="text-brand-blue dark:text-blue-400 hover:underline">
              Terms of Service
            </a>{' '}
            and{' '}
            <a href="#" className="text-brand-blue dark:text-blue-400 hover:underline">
              Privacy Policy
            </a>
          </p>

          {/* Sign In CTA */}
          <div className="mt-6 text-center">
            <span className="text-sm text-zinc-500 dark:text-zinc-400">
              Already have an account?{' '}
            </span>
            <a
              href="/"
              className="text-sm text-brand-blue dark:text-blue-400 font-bold hover:underline"
            >
              Sign In
            </a>
          </div>
        </div>
      </div>
    </main>
  );
}
