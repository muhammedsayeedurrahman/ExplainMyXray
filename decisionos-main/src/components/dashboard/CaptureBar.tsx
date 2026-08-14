'use client';

import React from 'react';
import { Accent, ACCENT_STYLES } from '@/config/roles';
import { VoiceState } from '@/hooks/useWorkspace';

interface CaptureBarProps {
  accent: Accent;
  placeholder: string;
  alertMsg: string | null;
  voiceState: VoiceState;
  transcribedText: string;
  textDirective: string;
  setTextDirective: (v: string) => void;
  onMicClick: () => void;
  onApplyVoiceDirective: () => void;
  onStructureText: (e: React.FormEvent) => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function CaptureBar({
  accent,
  placeholder,
  alertMsg,
  voiceState,
  transcribedText,
  textDirective,
  setTextDirective,
  onMicClick,
  onApplyVoiceDirective,
  onStructureText,
  onFileUpload,
}: CaptureBarProps) {
  const a = ACCENT_STYLES[accent];

  return (
    <div className="absolute bottom-0 left-0 right-0 px-6 pb-5 pt-3 bg-gradient-to-t from-zinc-50 via-zinc-50/95 dark:from-zinc-950 dark:via-zinc-950/95 to-transparent pointer-events-none">
      <div className="pointer-events-auto max-w-3xl mx-auto">
        {voiceState === 'done' && transcribedText && (
          <div className="mb-2 p-2.5 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl text-xs font-mono text-zinc-700 dark:text-zinc-300 shadow-sm flex items-center justify-between gap-2">
            <span className="break-words whitespace-normal flex-1">{transcribedText}</span>
            <button
              onClick={onApplyVoiceDirective}
              className={`shrink-0 ${a.solidBg} ${a.solidText} px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-wider cursor-pointer ${a.solidHoverBg} transition-colors`}
            >
              Apply &amp; Distribute
            </button>
          </div>
        )}

        {alertMsg && (
          <div className="mb-2 p-2.5 bg-zinc-900 dark:bg-zinc-800 text-white font-mono text-xs rounded-xl border border-zinc-700 shadow-md">
            &gt; {alertMsg}
          </div>
        )}

        <form
          onSubmit={onStructureText}
          className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.12)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.5)] overflow-hidden"
        >
          <textarea
            value={textDirective}
            onChange={(e) => setTextDirective(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (textDirective.trim()) onStructureText(e as unknown as React.FormEvent);
              }
            }}
            placeholder={placeholder}
            rows={2}
            className="w-full px-4 pt-3 pb-1 font-mono text-sm text-zinc-900 dark:text-white bg-transparent resize-none focus:outline-none placeholder:text-zinc-400 placeholder:text-xs"
          />
          <div className="flex items-center justify-between px-3 pb-3 gap-2">
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={onMicClick}
                title="Speak a directive"
                aria-label="Speak a directive"
                className={`p-2 rounded-xl transition-all cursor-pointer ${
                  voiceState === 'recording'
                    ? `${a.solidBg} ${a.solidText} recording-pulse`
                    : voiceState === 'thinking'
                    ? `${a.softBg} ${a.softText}`
                    : `text-zinc-400 ${a.softHoverBg}`
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 256 256" aria-hidden="true">
                  <path d="M128,176a48.05,48.05,0,0,0,48-48V64a48,48,0,0,0-96,0v64A48.05,48.05,0,0,0,128,176ZM96,64a32,32,0,0,1,64,0v64a32,32,0,0,1-64,0Zm40,143.6V232a8,8,0,0,1-16,0V207.6A80.11,80.11,0,0,1,48,128a8,8,0,0,1,16,0,64,64,0,0,0,128,0,8,8,0,0,1,16,0A80.11,80.11,0,0,1,136,207.6Z"/>
                </svg>
              </button>

              <label
                htmlFor="floating-file-upload"
                title="Upload image, PDF or document"
                className={`p-2 rounded-xl text-zinc-400 ${a.softHoverBg} cursor-pointer transition-all`}
              >
                <span className="sr-only">Upload image, PDF or document</span>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 256 256" aria-hidden="true">
                  <path d="M209.66,122.34a8,8,0,0,1,0,11.32l-82.05,82a56,56,0,0,1-79.2-79.21L147.67,36.73a40,40,0,1,1,56.61,56.55L105,193.94a24,24,0,0,1-33.94-33.94l82.76-84.05a8,8,0,1,1,11.46,11.16l-82.75,84a8,8,0,0,0,11.31,11.31l98.6-100.66a24,24,0,0,0-33.95-33.94L59.32,148a40,40,0,0,0,56.57,56.57l82.06-82A8,8,0,0,1,209.66,122.34Z"/>
                </svg>
                <input id="floating-file-upload" type="file" accept="image/*,.pdf,.doc,.docx,.xls,.xlsx" className="hidden" onChange={onFileUpload} />
              </label>

              {voiceState !== 'idle' && (
                <span className={`text-[10px] font-mono font-bold uppercase tracking-wider ${
                  voiceState === 'recording' ? a.text : 'text-zinc-400'
                }`}>
                  {voiceState === 'recording' && '● REC'}
                  {voiceState === 'thinking' && '⟳ AI...'}
                  {voiceState === 'done' && '✓ DONE'}
                </span>
              )}
            </div>

            <button
              type="submit"
              disabled={!textDirective.trim()}
              className={`flex items-center gap-1.5 ${a.solidBg} ${a.solidHoverBg} disabled:opacity-30 disabled:cursor-not-allowed ${a.solidText} px-3 sm:px-4 py-2 rounded-xl text-xs font-mono font-black uppercase tracking-wider transition-all cursor-pointer shadow-sm`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256">
                <path d="M224.49,136.49l-72,72a12,12,0,0,1-17-17L187,140H40a12,12,0,0,1,0-24H187L135.51,64.48a12,12,0,0,1,17-17l72,72A12,12,0,0,1,224.49,136.49Z"/>
              </svg>
              <span className="hidden min-[400px]:inline">Structure It</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
