import type { Metadata } from 'next';
import { Inter, DM_Serif_Display, DM_Sans } from 'next/font/google';
import './globals.css';
import { QueryProvider } from '@/components/providers/QueryProvider';
import { ChatWidget } from '@/components/chat/ChatWidget';
import { Navbar } from '@/components/navigation/Navbar';
import { Footer } from '@/components/navigation/Footer';
import { PageTransition } from '@/components/motion/PageTransition';
import { LoadingOverlay } from '@/components/ui/LoadingOverlay';
import { ScrollToTop } from '@/components/ui/ScrollToTop';
import { Toaster } from 'sonner';

const dmSerifDisplay = DM_Serif_Display({
  subsets: ['latin'],
  weight: '400',
  variable: '--font-display',
});

const dmSans = DM_Sans({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600'],
  variable: '--font-sans',
});

const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'AERUX',
  description: 'AERUX - Deep Learning Medical Analysis',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${dmSerifDisplay.variable} ${dmSans.variable} ${inter.variable} min-h-screen bg-[var(--color-surface-1)] text-[var(--color-aerux-navy)]`} suppressHydrationWarning>
        <QueryProvider>
          <div className="relative min-h-screen">
            <Navbar />
            <div className="relative">
              <PageTransition>
                {children}
              </PageTransition>
            </div>
            <Toaster richColors position="top-right" />
            <ScrollToTop />
            <LoadingOverlay />
            <Footer />
            <ChatWidget />
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
