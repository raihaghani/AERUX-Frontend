"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface LocationBarProps {
    label: string;
    value: number;   // 0–1
    rank?: number;
}

function barColor(v: number): string {
    if (v >= 0.65) return "#EF4444";
    if (v >= 0.35) return "#F59E0B";
    return "#3B82F6";
}

export function LocationBar({ label, value, rank }: LocationBarProps) {
    const [animated, setAnimated] = useState(false);

    useEffect(() => {
        const t = setTimeout(() => setAnimated(true), 120);
        return () => clearTimeout(t);
    }, [value]);

    const width = animated ? Math.round(value * 100) : 0;

    return (
        <div className="flex items-center gap-3">
            {rank !== undefined && (
                <span
                    className="text-[11px] font-semibold text-gray-400 w-4 shrink-0 tabular-nums"
                    style={{ fontFamily: "var(--font-display)" }}
                >
                    {rank}
                </span>
            )}
            <span className="text-[13px] text-gray-700 font-medium w-28 shrink-0 truncate">
                {label}
            </span>
            <div className="flex-1 h-[6px] rounded-full bg-[var(--color-surface-2)] overflow-hidden">
                <div
                    style={{
                        height: "100%",
                        width: `${width}%`,
                        background: barColor(value),
                        borderRadius: 3,
                        transition: "width 0.85s cubic-bezier(0.4,0,0.2,1), background 0.6s ease",
                    }}
                />
            </div>
            <span
                className="text-[12px] font-semibold tabular-nums w-10 text-right shrink-0"
                style={{ color: barColor(value), fontFamily: "var(--font-display)" }}
            >
                {Math.round(value * 100)}%
            </span>
        </div>
    );
}