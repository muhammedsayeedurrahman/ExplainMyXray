'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Sun, Moon } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'password' | 'otp'>('password');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [demoLoading, setDemoLoading] = useState<string | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

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

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const { signIn } = await import('@/lib/supabase/auth');
      await signIn({ email, password });

      // Redirect to appropriate dashboard based on user role
      // The middleware will handle redirection if user is already authenticated
      router.push('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign in failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSendOTP = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setError('OTP services are coming soon. Please use email & password for now.');
    }, 1000);
  };

  const handleDemoLogin = (role: string, targetPath: string) => {
    setError('');
    setDemoLoading(role);
    setTimeout(() => {
      router.push(targetPath);
    }, 600);
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
            THE OPERATIONAL BRAIN FOR FOUNDER-LED SMES
          </div>
          <h1 className="text-5xl lg:text-6xl font-logo font-black uppercase tracking-tight leading-[1.05] text-white">
            SPEAK THE<br />
            DECISION.<br />
            <span className="text-brand-red">WE RUN</span> THE<br />
            COMPANY.
          </h1>
          <p className="text-zinc-400 text-base font-normal leading-relaxed">
            Tailored to your industry — DecisionOS turns spoken directives into structured tasks, workflows and a shared operational brain.
          </p>
        </div>

        {/* Footer info */}
        <div className="flex items-center gap-2 z-10 text-zinc-500 dark:text-zinc-400 font-mono text-xs">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="text-brand-red animate-pulse" viewBox="0 0 256 256">
            <path d="M128,176a48.05,48.05,0,0,0,48-48V64a48,48,0,0,0-96,0v64A48.05,48.05,0,0,0,128,176ZM96,64a32,32,0,0,1,64,0v64a32,32,0,0,1-64,0Zm112,64a8,8,0,0,1-16,0,64,64,0,0,0-128,0,8,8,0,0,1-16,0,80.09,80.09,0,0,0,72,79.52V224h-32a8,8,0,0,1,0-16h80a8,8,0,0,1,0,16H136v16.48A80.09,80.09,0,0,0,208,128Z"/>
          </svg>
          <span>Voice-first &middot; AI-structured &middot; Multi-tenant</span>
        </div>
      </div>

      {/* Right Panel (Sign-in form) */}
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

        {/* Sign-in Form Container */}
        <div className="w-full max-w-[400px] flex flex-col justify-center animate-fade-up">
          
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
            SIGN IN
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-8 font-sans">
            Access your company brain.
          </p>

          {/* Authentication Method Tabs */}
          <div className="flex border border-zinc-950 dark:border-zinc-800 mb-6 bg-white dark:bg-zinc-950">
            <button
              onClick={() => { setActiveTab('password'); setError(''); }}
              className={`flex-1 py-3.5 text-xs font-mono font-bold uppercase tracking-wider transition-colors border-r border-zinc-950 dark:border-zinc-800 cursor-pointer ${
                activeTab === 'password'
                  ? 'bg-zinc-950 dark:bg-zinc-900 text-white border-zinc-950 dark:border-zinc-800'
                  : 'bg-white dark:bg-zinc-950 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              Email &amp; Password
            </button>
            <button
              onClick={() => { setActiveTab('otp'); setError(''); }}
              className={`flex-1 py-3.5 text-xs font-mono font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5 cursor-pointer ${
                activeTab === 'otp'
                  ? 'bg-zinc-950 dark:bg-zinc-900 text-white border-zinc-950 dark:border-zinc-800'
                  : 'bg-white dark:bg-zinc-950 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256">
                <path d="M176,16H80A24,24,0,0,0,56,40V216a24,24,0,0,0,24,24h96a24,24,0,0,0,24-24V40A24,24,0,0,0,176,16Zm8,200a8,8,0,0,1-8,8H80a8,8,0,0,1-8-8V40a8,8,0,0,1,8-8h96a8,8,0,0,1,8,8ZM128,188a12,12,0,1,1-12-12A12,12,0,0,1,128,188Z"/>
              </svg>
              Mobile OTP
            </button>
          </div>

          {/* Alert messages */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-950/20 border border-brand-red text-xs text-brand-red font-semibold rounded-md">
              {error}
            </div>
          )}

          {/* Email Form */}
          {activeTab === 'password' && (
            <form onSubmit={handleSignIn} className="space-y-4">
              <div>
                <label htmlFor="login-email" className="sr-only">
                  Email Address
                </label>
                <input
                  id="login-email"
                  type="email"
                  placeholder="Email"
                  required
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
                />
              </div>
              <div>
                <label htmlFor="login-password" className="sr-only">
                  Password
                </label>
                <input
                  id="login-password"
                  type="password"
                  placeholder="Password"
                  required
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-brand-red text-white font-bold uppercase tracking-wider py-3.5 border border-zinc-950 dark:border-zinc-800 hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] dark:hover:shadow-[3px_3px_0px_0px_rgba(255,59,48,0.4)] transition-all cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2 rounded-md"
              >
                {loading ? 'Signing in...' : 'Sign in'}
              </button>
            </form>
          )}

          {/* OTP Form */}
          {activeTab === 'otp' && (
            <form onSubmit={handleSendOTP} className="space-y-4">
              <div>
                <label htmlFor="login-phone" className="label-mono mb-2 text-zinc-400 dark:text-zinc-500">
                  Mobile Number
                </label>
                <input
                  id="login-phone"
                  type="tel"
                  placeholder="Registered mobile number"
                  required
                  autoComplete="tel"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  className="w-full border border-zinc-950 dark:border-zinc-800 px-4 py-3.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-brand-red/20 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 rounded-md"
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-brand-red text-white font-bold uppercase tracking-wider py-3.5 border border-zinc-950 dark:border-zinc-800 hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] dark:hover:shadow-[3px_3px_0px_0px_rgba(255,59,48,0.4)] transition-all cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2 rounded-md"
              >
                {loading ? 'Sending OTP...' : 'Send OTP'}
              </button>
            </form>
          )}

          {/* Signup CTA Link */}
          <div className="mt-4">
            <a
              href="/signup"
              className="text-sm text-brand-blue dark:text-blue-400 font-bold hover:underline"
            >
              Need a workspace? Register &rarr;
            </a>
          </div>

          {/* Quick-login demo roles */}
          <div className="mt-10 border-t border-zinc-200 dark:border-zinc-800 pt-8">
            <p className="label-mono mb-4 text-zinc-400 dark:text-zinc-500">
              Try the Sharma demo
            </p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { role: 'Owner', path: '/demo/owner' },
                { role: 'Sales', path: '/demo/sales' },
                { role: 'Production', path: '/demo/production' },
                { role: 'Finance', path: '/demo/finance' }
              ].map((demo) => (
                <button
                  key={demo.role}
                  onClick={() => handleDemoLogin(demo.role, demo.path)}
                  disabled={demoLoading !== null}
                  className="border border-zinc-950 dark:border-zinc-800 px-4 py-3 text-xs font-mono font-bold uppercase tracking-wider hover:bg-zinc-950 dark:hover:bg-zinc-800 hover:text-white dark:hover:text-white transition-colors cursor-pointer disabled:opacity-50 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-200 rounded-md"
                >
                  {demoLoading === demo.role ? 'Loading...' : demo.role}
                </button>
              ))}
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}
