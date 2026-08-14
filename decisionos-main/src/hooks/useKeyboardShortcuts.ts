import { useEffect, useCallback } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean; // Cmd on Mac, Win on Windows
  description: string;
  handler: () => void;
  enabled?: boolean;
}

/**
 * Custom hook for managing global keyboard shortcuts
 *
 * Usage:
 * ```tsx
 * useKeyboardShortcuts([
 *   {
 *     key: 'k',
 *     meta: true,
 *     description: 'Open command palette',
 *     handler: () => setCommandPaletteOpen(true),
 *   },
 * ]);
 * ```
 */
export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      const target = event.target as HTMLElement;
      const isEditing =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.contentEditable === 'true';

      for (const shortcut of shortcuts) {
        // Skip if shortcut is disabled
        if (shortcut.enabled === false) continue;

        // Check if key matches
        const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
        if (!keyMatches) continue;

        // Check modifiers
        const ctrlMatches = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
        const shiftMatches = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatches = shortcut.alt ? event.altKey : !event.altKey;
        const metaMatches = shortcut.meta ? event.metaKey : !event.metaKey;

        if (ctrlMatches && shiftMatches && altMatches && metaMatches) {
          // Special case: Allow Cmd+K even when editing (common UX pattern)
          const isCommandK = shortcut.meta && shortcut.key === 'k';

          if (!isEditing || isCommandK) {
            event.preventDefault();
            shortcut.handler();
            break;
          }
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

/**
 * Format keyboard shortcut for display
 *
 * @example
 * formatShortcut({ key: 'k', meta: true }) // "⌘K" on Mac, "Ctrl+K" on Windows
 */
export function formatShortcut(shortcut: Omit<KeyboardShortcut, 'description' | 'handler'>): string {
  const isMac = typeof navigator !== 'undefined' && navigator.platform.toUpperCase().indexOf('MAC') >= 0;

  const parts: string[] = [];

  if (shortcut.ctrl) parts.push(isMac ? '⌃' : 'Ctrl');
  if (shortcut.alt) parts.push(isMac ? '⌥' : 'Alt');
  if (shortcut.shift) parts.push(isMac ? '⇧' : 'Shift');
  if (shortcut.meta) parts.push(isMac ? '⌘' : 'Win');

  parts.push(shortcut.key.toUpperCase());

  return isMac ? parts.join('') : parts.join('+');
}

/**
 * Get modifier key symbol for current platform
 */
export function getModifierKey(): '⌘' | 'Ctrl' {
  const isMac = typeof navigator !== 'undefined' && navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  return isMac ? '⌘' : 'Ctrl';
}
