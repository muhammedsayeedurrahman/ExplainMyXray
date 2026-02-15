import createMiddleware from 'next-intl/middleware';
import { routing } from './i18n/routing';

export default createMiddleware(routing);

export const config = {
  matcher: ['/', '/(en|hi|ta|te|bn|mr|gu|kn|ml|pa|or)/:path*', '/((?!api|_next|_vercel|.*\\..*).*)'],
};
