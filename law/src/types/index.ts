export interface ThreeLayerAnswer {
  simple: string;
  practical: string;
  legal: string;
}

export interface ActionStep {
  step: number;
  title: string;
  description: string;
}

export interface KnowledgeEntry {
  id: string;
  categoryId: string;
  question: string;
  keywords: string[];
  answer: ThreeLayerAnswer;
  actionSteps: ActionStep[];
  relatedIds: string[];
}

export interface Category {
  id: string;
  icon: string;
  color: string;
  topicCount: number;
}

export interface EmergencyContact {
  name: string;
  number: string;
  description: string;
  available24x7: boolean;
  tollFree: boolean;
}

export interface StateLegalService {
  state: string;
  authority: string;
  phone: string;
  address: string;
  website?: string;
}

export interface EmergencyGuideStep {
  title: string;
  description: string;
}

export interface EmergencyGuide {
  id: string;
  title: string;
  description: string;
  steps: EmergencyGuideStep[];
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  results?: KnowledgeEntry[];
}

export interface SearchResult {
  item: KnowledgeEntry;
  score: number;
}
