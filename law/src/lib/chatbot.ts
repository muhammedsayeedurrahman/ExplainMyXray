import Fuse, { IFuseOptions } from 'fuse.js';
import { KnowledgeEntry, SearchResult } from '@/types';

let fuseInstance: Fuse<KnowledgeEntry> | null = null;
let indexedLocale: string | null = null;

const fuseOptions: IFuseOptions<KnowledgeEntry> = {
  keys: [
    { name: 'question', weight: 0.4 },
    { name: 'keywords', weight: 0.35 },
    { name: 'answer.simple', weight: 0.15 },
    { name: 'categoryId', weight: 0.1 },
  ],
  threshold: 0.4,
  includeScore: true,
  minMatchCharLength: 2,
  ignoreLocation: true,
};

export function initializeSearch(entries: KnowledgeEntry[], locale: string): void {
  fuseInstance = new Fuse(entries, fuseOptions);
  indexedLocale = locale;
}

export function searchKnowledgeBase(
  query: string,
  entries: KnowledgeEntry[],
  locale: string,
  limit: number = 5
): SearchResult[] {
  if (!fuseInstance || indexedLocale !== locale) {
    initializeSearch(entries, locale);
  }

  const results = fuseInstance!.search(query, { limit });

  return results.map((result) => ({
    item: result.item,
    score: result.score ?? 1,
  }));
}

export const suggestedQuestions = [
  'What should I do if police refuse to file my FIR?',
  'How can I file a consumer complaint?',
  'What are my rights if I am arrested?',
  'How do I report online fraud?',
  'Can women inherit property equally?',
  'What is the minimum wage in India?',
  'How to get free legal aid?',
  'What to do about workplace harassment?',
];
