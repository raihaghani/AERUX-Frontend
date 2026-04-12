"use client";

import { cn } from "@/lib/utils";

interface ScanOverlayProps {
    active: boolean;
    className?: string;
}

export function ScanOverlay({ active, className }: ScanOverlayProps) {
    if (!active) return null;

    return (
        <div className={cn("scan-overlay", className)}>
            <div className="flex flex-col items-center gap-3 z-10 relative">
                {/* Pulse ring */}
                <div className="w-10 h-10 rounded-full border-2 border-[var(--color-aerux-accent)]
                        border-t-transparent animate-spin opacity-80" />
                <p className="text-white/90 text-[12px] tracking-[0.12em] uppercase font-medium">
                    Analysing
                </p>
            </div>
        </div>
    );
}