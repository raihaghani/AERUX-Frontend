"use client";

import { useEffect, useState } from "react";

interface RiskGaugeProps {
    probability: number;       // 0–1
    riskLevel: string;
    size?: number;
}

function gaugeColor(riskLevel: string): string {
    const map: Record<string, string> = {
        "Minimal": "#22C55E",
        "Low": "#3B82F6",
        "Moderate": "#F59E0B",
        "High": "#EF4444",
        "Critical": "#DC2626",
    };
    return map[riskLevel] ?? "#3B82F6";
}

export function RiskGauge({ probability = 0, riskLevel = "Minimal", size = 96 }: RiskGaugeProps) {
    const [animated, setAnimated] = useState(false);
    const r = (size / 2) - 8;
    const circ = 2 * Math.PI * r;
    const color = gaugeColor(riskLevel);
    const cx = size / 2;
    const cy = size / 2;

    useEffect(() => {
        const t = setTimeout(() => setAnimated(true), 100);
        return () => clearTimeout(t);
    }, [probability]);

    const fill = animated ? Math.max(0, Math.min(1, probability)) * circ : 0;

    return (
        <div className="relative" style={{ width: size, height: size, flexShrink: 0 }}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {/* Track */}
                <circle cx={cx} cy={cy} r={r} fill="none" stroke="#E8EEF6" strokeWidth="8" />

                {/* Glow halo */}
                <circle
                    cx={cx} cy={cy} r={r} fill="none"
                    stroke={color} strokeWidth="16" opacity="0.09"
                    strokeDasharray={`${fill} ${circ}`}
                    strokeLinecap="round"
                    transform={`rotate(-90 ${cx} ${cy})`}
                    style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)" }}
                />

                {/* Main arc */}
                <circle
                    cx={cx} cy={cy} r={r} fill="none"
                    stroke={color} strokeWidth="8"
                    strokeDasharray={`${fill} ${circ}`}
                    strokeLinecap="round"
                    transform={`rotate(-90 ${cx} ${cy})`}
                    style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)" }}
                />
            </svg>

            {/* Inner label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span
                    className="text-[21px] font-bold leading-none text-[var(--color-aerux-navy)]"
                    style={{ fontFamily: "var(--font-display)" }}
                >
                    {Math.round(probability * 100)}%
                </span>
                <span className="text-[9px] tracking-[0.12em] uppercase text-gray-400 mt-1">
                    Probability
                </span>
            </div>
        </div>
    );
}