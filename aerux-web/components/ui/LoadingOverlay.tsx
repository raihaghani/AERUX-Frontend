"use client";

import { useUIStore } from "@/store/ui";
import { AnimatePresence, motion } from "framer-motion";

export function LoadingOverlay() {
    const isLoading = useUIStore((s) => s.isLoading);

    return (
        <AnimatePresence>
            {isLoading && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="fixed inset-0 z-[100] flex flex-col items-center justify-center
                     bg-white/90 backdrop-blur-sm"
                >
                    {/* Animated rings */}
                    <div className="relative w-16 h-16 mb-5">
                        <div className="absolute inset-0 rounded-full border-2 border-[var(--color-aerux-accent)]/20
                            animate-ping" />
                        <div className="absolute inset-1 rounded-full border-2 border-[var(--color-aerux-accent)]/40
                            animate-ping [animation-delay:0.3s]" />
                        <div className="absolute inset-3 rounded-full border-2 border-t-[var(--color-aerux-accent)]
                            border-[var(--color-aerux-accent)]/20 animate-spin" />
                    </div>

                    <p
                        className="text-[var(--color-aerux-navy)] text-[15px] font-semibold tracking-wide"
                        style={{ fontFamily: "var(--font-sans)" }}
                    >
                        Processing Scan
                    </p>
                    <p className="text-gray-400 text-[12px] tracking-[0.1em] uppercase mt-1">
                        AI Pipeline Running
                    </p>
                </motion.div>
            )}
        </AnimatePresence>
    );
}