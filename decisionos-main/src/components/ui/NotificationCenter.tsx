'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, CheckSquare, AtSign, Clock, ArrowRight, Settings, X, Check } from 'lucide-react';

export type NotificationType = 'task_assigned' | 'mention' | 'deadline' | 'handoff' | 'comment';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  actionUrl?: string;
  actor?: {
    name: string;
    avatar?: string;
  };
}

interface NotificationCenterProps {
  notifications?: Notification[];
  onNotificationClick?: (notification: Notification) => void;
  onMarkAsRead?: (notificationId: string) => void;
  onMarkAllAsRead?: () => void;
  onClearAll?: () => void;
  onOpenSettings?: () => void;
}

const NOTIFICATION_ICONS: Record<NotificationType, React.ReactNode> = {
  task_assigned: <CheckSquare className="w-4 h-4" />,
  mention: <AtSign className="w-4 h-4" />,
  deadline: <Clock className="w-4 h-4" />,
  handoff: <ArrowRight className="w-4 h-4" />,
  comment: <AtSign className="w-4 h-4" />,
};

const NOTIFICATION_COLORS: Record<NotificationType, string> = {
  task_assigned: 'text-brand-blue',
  mention: 'text-brand-yellow',
  deadline: 'text-brand-red',
  handoff: 'text-green-600',
  comment: 'text-zinc-600',
};

/**
 * Notification center with dropdown panel
 *
 * Displays notifications with filtering, marking as read, and actions
 *
 * Usage:
 * ```tsx
 * <NotificationCenter
 *   notifications={notifications}
 *   onNotificationClick={(notif) => router.push(notif.actionUrl)}
 *   onMarkAsRead={(id) => markNotificationRead(id)}
 * />
 * ```
 */
export function NotificationCenter({
  notifications = [],
  onNotificationClick,
  onMarkAsRead,
  onMarkAllAsRead,
  onClearAll,
  onOpenSettings,
}: NotificationCenterProps) {
  const [isOpen, setIsOpen] = useState(false);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleNotificationClick = (notification: Notification) => {
    if (onMarkAsRead && !notification.read) {
      onMarkAsRead(notification.id);
    }
    if (onNotificationClick) {
      onNotificationClick(notification);
    }
    setIsOpen(false);
  };

  const formatTimestamp = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="relative">
      {/* Bell Icon Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="
          relative
          p-2
          rounded-lg
          hover:bg-zinc-100 dark:hover:bg-zinc-800
          transition-colors
        "
        aria-label="Notifications"
      >
        <Bell className="w-5 h-5 text-zinc-700 dark:text-zinc-300" />
        {unreadCount > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="
              absolute -top-1 -right-1
              w-5 h-5
              bg-brand-red
              text-white text-xs font-bold
              rounded-full
              flex items-center justify-center
              border-2 border-white dark:border-zinc-900
            "
          >
            {unreadCount > 9 ? '9+' : unreadCount}
          </motion.span>
        )}
      </button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="
                absolute right-0 top-full mt-2
                w-96 max-w-[calc(100vw-2rem)]
                bg-white dark:bg-zinc-900
                rounded-xl
                shadow-xl
                border border-zinc-200 dark:border-zinc-800
                overflow-hidden
                z-50
              "
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-800">
                <h3 className="font-bold text-zinc-900 dark:text-zinc-100">
                  Notifications
                </h3>
                <div className="flex items-center gap-2">
                  {notifications.length > 0 && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onMarkAllAsRead?.();
                        }}
                        className="text-xs text-brand-red hover:underline font-medium"
                        title="Mark all as read"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onOpenSettings?.();
                          setIsOpen(false);
                        }}
                        className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100"
                        title="Notification settings"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                    </>
                  )}
                  <button
                    onClick={() => setIsOpen(false)}
                    className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Notification List */}
              <div className="max-h-[400px] overflow-y-auto">
                {notifications.length === 0 ? (
                  <div className="p-8 text-center">
                    <Bell className="w-12 h-12 text-zinc-300 dark:text-zinc-700 mx-auto mb-3" />
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">
                      You're all caught up!
                    </p>
                  </div>
                ) : (
                  <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
                    {notifications.map((notification) => (
                      <motion.button
                        key={notification.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => handleNotificationClick(notification)}
                        className={`
                          w-full p-4 text-left
                          hover:bg-zinc-50 dark:hover:bg-zinc-800
                          transition-colors
                          ${!notification.read ? 'bg-brand-red/5 dark:bg-brand-red/10' : ''}
                        `}
                      >
                        <div className="flex items-start gap-3">
                          {/* Icon */}
                          <div className={`
                            flex-shrink-0 w-8 h-8
                            flex items-center justify-center
                            rounded-full
                            bg-zinc-100 dark:bg-zinc-800
                            ${NOTIFICATION_COLORS[notification.type]}
                          `}>
                            {NOTIFICATION_ICONS[notification.type]}
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between gap-2 mb-1">
                              <p className="font-medium text-sm text-zinc-900 dark:text-zinc-100">
                                {notification.title}
                              </p>
                              {!notification.read && (
                                <div className="w-2 h-2 bg-brand-red rounded-full flex-shrink-0 mt-1" />
                              )}
                            </div>
                            <p className="text-xs text-zinc-600 dark:text-zinc-400 line-clamp-2 mb-1">
                              {notification.message}
                            </p>
                            <div className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-500">
                              {notification.actor && (
                                <span>{notification.actor.name}</span>
                              )}
                              <span>•</span>
                              <span>{formatTimestamp(notification.timestamp)}</span>
                            </div>
                          </div>
                        </div>
                      </motion.button>
                    ))}
                  </div>
                )}
              </div>

              {/* Footer */}
              {notifications.length > 0 && (
                <div className="p-3 border-t border-zinc-200 dark:border-zinc-800">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onClearAll?.();
                    }}
                    className="
                      w-full
                      text-xs font-medium
                      text-zinc-600 dark:text-zinc-400
                      hover:text-brand-red dark:hover:text-brand-red
                      transition-colors
                    "
                  >
                    Clear all notifications
                  </button>
                </div>
              )}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
