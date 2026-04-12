"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
    { label: "Upload Scan", href: "/upload", icon: "⊕" },
    { label: "Visuals", href: "/visuals", icon: "⊞" },
    { label: "Reports", href: "/reports", icon: "≡" },
] as const;

export function Navbar() {
    const pathname = usePathname();

    return (
        <header
            className={cn(
                "fixed top-0 z-50 w-full h-[62px] flex items-center justify-between px-6 bg-transparent"
            )}
        >
            {/* Logo — left */}
            <Link href="/" className="flex items-center gap-2.5 shrink-0 group">
                <svg width="26" height="26" viewBox="0 0 22 22" fill="none" aria-hidden>
                    <circle cx="11" cy="11" r="10" stroke="#0074D9" strokeWidth="1.5" />
                    <path d="M6 11 Q11 5 16 11 Q11 17 6 11Z" fill="#0074D9" opacity=".18" />
                    <circle cx="11" cy="11" r="3.5" fill="#0074D9" />
                </svg>
                <span
                    className="text-[22px] uppercase font-bold tracking-[0.04em] text-[var(--color-aerux-navy)]"
                    style={{ fontFamily: "var(--font-display)" }}
                >
                    Aerux
                </span>
            </Link>

            {/* Pill nav — centre, absolutely positioned */}
            <nav
                className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2
                   bg-[rgba(240,243,250,0.9)] border border-[rgba(221,230,240,0.8)]
                   rounded-full px-2 py-1.5"
            >
                {NAV_ITEMS.map((item) => {
                    const active = pathname === item.href ||
                        pathname.startsWith(item.href);
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "flex items-center gap-2 px-5 py-2.5 rounded-full text-[15px]",
                                "font-medium whitespace-nowrap transition-all duration-150",
                                "font-[var(--font-sans)]",
                                active
                                    ? "bg-white text-[var(--color-aerux-navy)] font-semibold shadow-[0_1px_6px_rgba(10,20,72,0.08),0_0_0_0.5px_rgba(221,230,240,0.9)]"
                                    : "text-gray-500 hover:bg-white/90 hover:text-[var(--color-aerux-navy)] hover:-translate-y-px"
                            )}
                        >
                            <span className={cn("text-[14px]", active ? "opacity-100 text-[var(--color-aerux-accent)]" : "opacity-50")}>
                                {item.icon}
                            </span>
                            {item.label}
                        </Link>
                    );
                })}
            </nav>

            {/* CTA — right */}
            <div className="flex items-center gap-2.5 shrink-0">
                {/* About Button */}
                <Link
                    href="/about"
                    className="flex items-center gap-1.5 px-4 py-2 rounded-full
                     bg-[var(--color-aerux-navy)] text-white text-[13px] font-semibold
                     tracking-[0.02em] transition-all duration-200 group
                     hover:bg-[#0d1c5e] hover:-translate-y-px
                     hover:shadow-[0_4px_14px_rgba(10,20,72,0.25)]
                     active:translate-y-0 active:shadow-none"
                >
                    About
                    <span className="text-[12px] opacity-70 transition-transform duration-200 group-hover:translate-x-px group-hover:-translate-y-px">
                        ◎
                    </span>
                </Link>
            </div>
        </header>
    );
}
