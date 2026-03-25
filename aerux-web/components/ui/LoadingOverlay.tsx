"use client";

import { useUIStore } from "@/store/ui";

export function LoadingOverlay() {
  const isLoading = useUIStore((s) => s.isLoading);
  if (!isLoading) return null;
  return (
    <div className="fixed inset-0 z-[70] grid place-items-center bg-white/60 backdrop-blur-sm">
      <div className="h-14 w-14 animate-pulse rounded-full border-4 border-[color:var(--aerux-navy)]/20 border-t-[color:var(--aerux-navy)]" style={{ borderTopWidth: 4 }} />
      <span className="sr-only">Loading</span>
    </div>
  );
}


