'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { getSharedState, saveSharedState, routeDirective, WorkspaceState, TaskCard, LOCAL_STORAGE_KEY } from '@/utils/sharedState';
import { Role, ROLES } from '@/config/roles';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { useFileUpload } from '@/hooks/useFileUpload';

export type VoiceState = 'idle' | 'recording' | 'thinking' | 'done';

const EMPTY_STATE: WorkspaceState = {
  cards: [],
  handoffs: [],
  notifications: { owner: 4, sales: 7, production: 3, finance: 5 },
};

const roleLabel = (role: Role) => role.charAt(0).toUpperCase() + role.slice(1);

/**
 * Everything a demo dashboard needs: theme, shared workspace state, the
 * decision-desk card mutations, the floating capture bar (voice/text/upload),
 * and toast feedback. Extracted so the four role pages (owner/sales/
 * production/finance) don't each hand-roll an identical copy.
 */
export function useWorkspace(role: Role) {
  const config = ROLES[role];

  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [workspaceState, setWorkspaceState] = useState<WorkspaceState>(EMPTY_STATE);
  const [alertMsg, setAlertMsg] = useState<string | null>(null);

  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [transcribedText, setTranscribedText] = useState('');
  const [textDirective, setTextDirective] = useState('');

  // Real audio recording and transcription
  const audioRecorder = useAudioRecorder();

  // Real file upload to Supabase Storage
  const fileUpload = useFileUpload();

  // Define triggerAlert early since it's used in effects below
  const triggerAlert = useCallback((msg: string) => {
    setAlertMsg(msg);
    setTimeout(() => setAlertMsg(null), 3000);
  }, []);

  const updateState = useCallback((next: WorkspaceState) => {
    setWorkspaceState(next);
    saveSharedState(next);
  }, []);

  useEffect(() => {
    setWorkspaceState(getSharedState());
  }, []);

  // Cross-tab sync: another role's tab writes to localStorage, this tab
  // picks it up via the native `storage` event (fires only in *other* tabs).
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key !== LOCAL_STORAGE_KEY || !e.newValue) return;
      try {
        setWorkspaceState(JSON.parse(e.newValue));
      } catch {
        // ignore malformed writes from another tab
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = savedTheme === 'dark' || (!savedTheme && prefersDark) ? 'dark' : 'light';
    setTheme(next);
    document.documentElement.classList.toggle('dark', next === 'dark');
  }, []);

  // Sync audio recorder state to voice state
  useEffect(() => {
    const recorderState = audioRecorder.state;
    if (recorderState === 'idle') {
      setVoiceState('idle');
    } else if (recorderState === 'recording') {
      setVoiceState('recording');
    } else if (recorderState === 'processing') {
      setVoiceState('thinking');
    } else if (recorderState === 'done') {
      setVoiceState('done');
    } else if (recorderState === 'error') {
      setVoiceState('idle');
      if (audioRecorder.error) {
        triggerAlert(audioRecorder.error);
      }
    }
  }, [audioRecorder.state, audioRecorder.error, triggerAlert]);

  // Sync transcription from audio recorder
  useEffect(() => {
    if (audioRecorder.transcription) {
      setTranscribedText(audioRecorder.transcription);
    }
  }, [audioRecorder.transcription]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'light' ? 'dark' : 'light';
      document.documentElement.classList.toggle('dark', next === 'dark');
      localStorage.setItem('theme', next);
      return next;
    });
  }, []);

  const handleMarkDone = useCallback((id: number) => {
    setWorkspaceState(prev => {
      const next = { ...prev, cards: prev.cards.map(c => c.id === id ? { ...c, done: !c.done } : c) };
      saveSharedState(next);
      return next;
    });
  }, []);

  const handleDismiss = useCallback((id: number) => {
    setWorkspaceState(prev => {
      const next = { ...prev, cards: prev.cards.filter(c => c.id !== id) };
      saveSharedState(next);
      return next;
    });
  }, []);

  const handleSendToBoard = useCallback((title: string) => {
    triggerAlert(`"${title}" advanced to Loom Workflows Board.`);
  }, [triggerAlert]);

  const handleClearNotifications = useCallback(() => {
    setWorkspaceState(prev => {
      const next = { ...prev, notifications: { ...prev.notifications, [role]: 0 } };
      saveSharedState(next);
      return next;
    });
    triggerAlert('Notifications cleared.');
  }, [role, triggerAlert]);

  const distributeCard = useCallback((card: TaskCard) => {
    setWorkspaceState(prev => {
      const nextNotifications = { ...prev.notifications };
      if (card.assignedTo !== role) {
        nextNotifications[card.assignedTo] += 1;
      }
      const next = { ...prev, cards: [card, ...prev.cards], notifications: nextNotifications };
      saveSharedState(next);
      return next;
    });
  }, [role]);

  const handleMicClick = useCallback(async () => {
    if (voiceState === 'idle') {
      // Start real audio recording
      await audioRecorder.startRecording();
    } else if (voiceState === 'recording') {
      // Stop recording and trigger transcription
      await audioRecorder.stopRecording();
    } else {
      // Cancel or reset
      audioRecorder.cancelRecording();
      setTranscribedText('');
    }
  }, [voiceState, audioRecorder]);

  const handleApplyVoiceDirective = useCallback(() => {
    if (!transcribedText) return;
    const parsed = routeDirective(transcribedText);
    distributeCard({
      id: Date.now(),
      title: 'Voice: ' + transcribedText.substring(0, 45) + '...',
      subtext: transcribedText,
      type: 'TASK',
      source: 'VOICE',
      category: parsed.category,
      assignedTo: parsed.assignedTo,
      done: false,
    });
    setVoiceState('idle');
    setTranscribedText('');
    triggerAlert(`Voice directive assigned to ${parsed.assignedTo.toUpperCase()} and distributed!`);
  }, [transcribedText, distributeCard, triggerAlert]);

  const handleStructureText = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!textDirective.trim()) return;
    const parsed = routeDirective(textDirective);
    distributeCard({
      id: Date.now(),
      title: textDirective,
      subtext: `${roleLabel(role)} instruction assigned to ${parsed.assignedTo.toUpperCase()}`,
      type: 'TASK',
      source: 'TEXT',
      category: parsed.category,
      assignedTo: parsed.assignedTo,
      done: false,
    });
    setTextDirective('');
    triggerAlert(`Structured and assigned to ${parsed.assignedTo.toUpperCase()} successfully.`);
  }, [textDirective, role, distributeCard, triggerAlert]);

  const handleFileUpload = useCallback(async (file: File) => {
    if (!file) return;

    try {
      triggerAlert(`Uploading ${file.name}...`);

      // Upload file to Supabase Storage
      const result = await fileUpload.upload(file, {
        bucket: 'documents',
        folder: 'uploads',
      });

      // Create a task card based on the uploaded file
      const parsed = routeDirective(`Process uploaded document: ${file.name}`);
      distributeCard({
        id: Date.now(),
        title: `Process uploaded file: ${file.name}`,
        subtext: `Uploaded to: ${result.path}`,
        type: file.type.includes('pdf') ? 'INVOICE' : 'TASK',
        source: 'UPLOAD',
        category: parsed.category,
        assignedTo: parsed.assignedTo,
        done: false,
      });

      triggerAlert(`${file.name} uploaded and assigned to ${parsed.assignedTo.toUpperCase()}.`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Upload failed';
      triggerAlert(`Upload failed: ${errorMsg}`);
    }
  }, [fileUpload, distributeCard, triggerAlert]);

  // Wrapper to extract file from input event
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
    // Reset input value so the same file can be uploaded again
    e.target.value = '';
  }, [handleFileUpload]);

  return {
    config,
    theme,
    toggleTheme,
    workspaceState,
    updateState,
    distributeCard,
    alertMsg,
    triggerAlert,
    voiceState,
    transcribedText,
    textDirective,
    setTextDirective,
    handleMicClick,
    handleApplyVoiceDirective,
    handleStructureText,
    handleFileUpload: handleFileInputChange,
    handleMarkDone,
    handleDismiss,
    handleSendToBoard,
    handleClearNotifications,
  };
}
