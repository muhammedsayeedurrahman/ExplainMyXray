import { HeroSection } from '@/components/home/HeroSection';
import { FeaturesSection } from '@/components/home/FeaturesSection';
import { QuickAccessCards } from '@/components/home/QuickAccessCards';
import { StatsSection } from '@/components/home/StatsSection';

export default function HomePage() {
  return (
    <>
      <HeroSection />
      <FeaturesSection />
      <StatsSection />
      <QuickAccessCards />
    </>
  );
}
