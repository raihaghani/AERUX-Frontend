import type { Metadata } from "next";
import { Inter, Poppins } from "next/font/google";
import "./globals.css";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { Navbar } from "@/components/navigation/Navbar";
import { Footer } from "@/components/navigation/Footer";
import { PageTransition } from "@/components/motion/PageTransition";
import { LoadingOverlay } from "@/components/ui/LoadingOverlay";
import { ScrollToTop } from "@/components/ui/ScrollToTop";
import { Toaster } from "sonner";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AERUX Dashboard",
  description: "Responsive medical AI dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`light ${inter.variable} ${poppins.variable}`}>
      <body className="min-h-screen bg-background text-foreground" suppressHydrationWarning>
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
