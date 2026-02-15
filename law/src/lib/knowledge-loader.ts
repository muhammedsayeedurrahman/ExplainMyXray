import { KnowledgeEntry } from '@/types';

const knowledgeCache = new Map<string, KnowledgeEntry[]>();

const categoryFiles = [
  'women-rights',
  'worker-rights',
  'consumer-rights',
  'police-fir',
  'property',
  'cyber-crime',
  'child-rights',
  'legal-aid',
] as const;

export async function loadKnowledgeBase(locale: string): Promise<KnowledgeEntry[]> {
  const cacheKey = locale;
  if (knowledgeCache.has(cacheKey)) {
    return knowledgeCache.get(cacheKey)!;
  }

  const results = await Promise.allSettled(
    categoryFiles.map(async (category) => {
      try {
        const data = await import(`@/data/knowledge-base/${locale}/${category}.json`);
        return (data.default || data) as KnowledgeEntry[];
      } catch {
        if (locale !== 'en') {
          const data = await import(`@/data/knowledge-base/en/${category}.json`);
          return (data.default || data) as KnowledgeEntry[];
        }
        return [] as KnowledgeEntry[];
      }
    })
  );

  const allEntries: KnowledgeEntry[] = results.flatMap((result) =>
    result.status === 'fulfilled' ? result.value : []
  );

  knowledgeCache.set(cacheKey, allEntries);
  return allEntries;
}

export async function loadCategoryEntries(
  locale: string,
  categoryId: string
): Promise<KnowledgeEntry[]> {
  const all = await loadKnowledgeBase(locale);
  return all.filter((entry) => entry.categoryId === categoryId);
}

export async function loadEntry(
  locale: string,
  entryId: string
): Promise<KnowledgeEntry | undefined> {
  const all = await loadKnowledgeBase(locale);
  return all.find((entry) => entry.id === entryId);
}

export async function getRelatedEntries(
  locale: string,
  relatedIds: string[]
): Promise<KnowledgeEntry[]> {
  const all = await loadKnowledgeBase(locale);
  return all.filter((entry) => relatedIds.includes(entry.id));
}
