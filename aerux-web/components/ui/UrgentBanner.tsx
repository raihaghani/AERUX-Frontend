"use client";

interface UrgentBannerProps {
    message: string;
    subtitle?: string;
}

export function UrgentBanner({ message, subtitle }: UrgentBannerProps) {
    return (
        <div
            className="flex items-start gap-3 p-4 rounded-xl
                 bg-red-50 border border-red-200/60
                 animate-[fadeUp_0.3s_ease_both,urgentRing_2.2s_ease_0.5s_3]"
        >
            <span className="text-red-500 mt-0.5 text-[18px] shrink-0">⚠</span>
            <div>
                <p className="text-[14px] font-semibold text-red-700">{message}</p>
                {subtitle && (
                    <p className="text-[13px] text-red-500/80 mt-0.5">{subtitle}</p>
                )}
            </div>
        </div>
    );
}