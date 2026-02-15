'use client';

import { ReactNode } from 'react';
import {
  motion,
  useReducedMotion,
  Variants,
} from 'framer-motion';

interface AnimationProps {
  children: ReactNode;
  className?: string;
  delay?: number;
}

const noMotion: Variants = {
  hidden: { opacity: 1 },
  visible: { opacity: 1 },
};

export function FadeIn({ children, className, delay = 0 }: AnimationProps) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      variants={
        reduced
          ? noMotion
          : {
              hidden: { opacity: 0 },
              visible: { opacity: 1, transition: { duration: 0.5, delay } },
            }
      }
    >
      {children}
    </motion.div>
  );
}

export function SlideUp({ children, className, delay = 0 }: AnimationProps) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      variants={
        reduced
          ? noMotion
          : {
              hidden: { opacity: 0, y: 30 },
              visible: {
                opacity: 1,
                y: 0,
                transition: { duration: 0.5, delay },
              },
            }
      }
    >
      {children}
    </motion.div>
  );
}

interface SlideInProps extends AnimationProps {
  direction?: 'left' | 'right';
}

export function SlideIn({
  children,
  className,
  delay = 0,
  direction = 'left',
}: SlideInProps) {
  const reduced = useReducedMotion();
  const x = direction === 'left' ? -30 : 30;
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      variants={
        reduced
          ? noMotion
          : {
              hidden: { opacity: 0, x },
              visible: {
                opacity: 1,
                x: 0,
                transition: { duration: 0.5, delay },
              },
            }
      }
    >
      {children}
    </motion.div>
  );
}

interface StaggerContainerProps {
  children: ReactNode;
  className?: string;
  staggerDelay?: number;
}

export function StaggerContainer({
  children,
  className,
  staggerDelay = 0.1,
}: StaggerContainerProps) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      variants={
        reduced
          ? { hidden: {}, visible: {} }
          : {
              hidden: {},
              visible: {
                transition: { staggerChildren: staggerDelay },
              },
            }
      }
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({ children, className }: { children: ReactNode; className?: string }) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      variants={
        reduced
          ? noMotion
          : {
              hidden: { opacity: 0, y: 20 },
              visible: {
                opacity: 1,
                y: 0,
                transition: { duration: 0.4 },
              },
            }
      }
    >
      {children}
    </motion.div>
  );
}

export function ScaleIn({ children, className, delay = 0 }: AnimationProps) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      variants={
        reduced
          ? noMotion
          : {
              hidden: { opacity: 0, scale: 0.9 },
              visible: {
                opacity: 1,
                scale: 1,
                transition: { duration: 0.4, delay },
              },
            }
      }
    >
      {children}
    </motion.div>
  );
}
