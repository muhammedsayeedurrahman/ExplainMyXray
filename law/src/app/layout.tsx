import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Know Your Rights India - Legal Rights Awareness',
  description:
    'Understanding your legal rights made simple. Learn about Indian laws, file complaints, and access emergency legal help in multiple languages.',
  keywords: ['Indian law', 'legal rights', 'FIR', 'consumer rights', 'women rights', 'legal aid India'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
