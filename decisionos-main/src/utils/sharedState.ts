'use client';

export interface TaskCard {
  id: number;
  title: string;
  subtext: string;
  type: 'TASK' | 'REMINDER' | 'INVOICE' | 'APPROVAL';
  source: 'TEXT' | 'VOICE' | 'UPLOAD';
  detailsCount?: number;
  category: 'CUSTOMER' | 'SUPPLIER' | 'INVOICE' | 'PAYMENT' | 'COMPLAINT' | 'OTHER';
  done?: boolean;
  assignedTo: 'owner' | 'sales' | 'production' | 'finance';
  // Explicit calendar placement chosen via "Add Task" (date input, ISO
  // 'YYYY-MM-DD'; time input, 24h 'HH:MM'). When present, TaskCalendarFeed's
  // scheduler uses these instead of guessing from the title.
  scheduledDate?: string;
  scheduledTime?: string;
}

export interface HandoffItem {
  id: 'sales_handoff' | 'production_handoff';
  title: string;
  description: string;
  instruction: string;
  status: 'pending' | 'submitted' | 'approved';
  replyText: string;
}

export interface WorkspaceState {
  cards: TaskCard[];
  handoffs: HandoffItem[];
  notifications: {
    owner: number;
    sales: number;
    production: number;
    finance: number;
  };
}

import { DEMO_CARDS, DEMO_HANDOFFS, DEMO_NOTIFICATIONS } from '@/fixtures/demo-data';

export const LOCAL_STORAGE_KEY = 'sharma_workspace_state';

// Use demo data in development, empty state in production
const isDevelopment = process.env.NODE_ENV === 'development';
const initialCards: TaskCard[] = isDevelopment ? DEMO_CARDS : [];
const initialHandoffs: HandoffItem[] = isDevelopment ? DEMO_HANDOFFS : [];
const DEFAULT_NOTIFICATIONS = isDevelopment ? DEMO_NOTIFICATIONS : { owner: 0, sales: 0, production: 0, finance: 0 };

export function getSharedState(): WorkspaceState {
  if (typeof window === 'undefined') {
    return {
      cards: initialCards,
      handoffs: initialHandoffs,
      notifications: { ...DEFAULT_NOTIFICATIONS }
    };
  }
  const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
  if (!raw) {
    const defaultState: WorkspaceState = {
      cards: initialCards,
      handoffs: initialHandoffs,
      notifications: { ...DEFAULT_NOTIFICATIONS }
    };
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(defaultState));
    return defaultState;
  }
  try {
    return JSON.parse(raw);
  } catch (e) {
    return {
      cards: initialCards,
      handoffs: initialHandoffs,
      notifications: { ...DEFAULT_NOTIFICATIONS }
    };
  }
}

export function saveSharedState(state: WorkspaceState) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(state));
}

const ROLE_LABELS: Record<TaskCard['assignedTo'], string> = {
  owner: 'Owner',
  sales: 'Sales',
  production: 'Production',
  finance: 'Finance',
};

// Keyword lists checked in priority order (sales, then production, then finance).
// Kept as one source of truth so routeDirective() and explainRouting() below
// can never drift apart and show a reason that doesn't match the assignment.
const ROUTING_KEYWORDS: [Exclude<TaskCard['assignedTo'], 'owner'>, TaskCard['category'], string[]][] = [
  ['sales', 'CUSTOMER', ['priya', 'sales', 'sell', 'retailer']],
  ['production', 'SUPPLIER', ['amit', 'production', 'loom', 'fabric', 'produce']],
  ['finance', 'INVOICE', ['sunita', 'finance', 'invoice', 'payment', 'pay', 'cost']],
];

// Helper to route directives by keywords
export function routeDirective(text: string): { assignedTo: TaskCard['assignedTo']; category: TaskCard['category']; reason: string } {
  const lower = text.toLowerCase();

  for (const [assignedTo, category, keywords] of ROUTING_KEYWORDS) {
    const matched = keywords.filter(k => lower.includes(k));
    if (matched.length > 0) {
      return {
        assignedTo,
        category,
        reason: `Matched "${matched.join('", "')}" → routed to ${ROLE_LABELS[assignedTo]}`,
      };
    }
  }
  return { assignedTo: 'owner', category: 'OTHER', reason: 'No routing keywords matched — held with Owner for review' };
}

// Explains why a card ended up with its current assignee, for display as a
// "routing rationale" chip. Cards created live by routeDirective() will always
// re-match their own text (it's a pure function of title+subtext), so no
// separate routingReason field needs to be stored or kept in sync.
export function explainRouting(card: Pick<TaskCard, 'title' | 'subtext' | 'assignedTo' | 'source'>): string {
  if (card.source === 'UPLOAD') {
    return `Detected from uploaded document → routed to ${ROLE_LABELS[card.assignedTo]}`;
  }
  const guess = routeDirective(`${card.title} ${card.subtext}`);
  if (guess.assignedTo === card.assignedTo) return guess.reason;
  return `Routed to ${ROLE_LABELS[card.assignedTo]}`;
}
