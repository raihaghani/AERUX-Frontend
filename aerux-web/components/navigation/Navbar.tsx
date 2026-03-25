"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Upload Scan" },
  { href: "/visuals", label: "Visuals" },
  { href: "/reports", label: "Reports" },
  { href: "/about", label: "About" },
  { href: "/contact", label: "Contact" },
];

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Prefetch common routes to speed up navigation
  useEffect(() => {
    LINKS.forEach((l) => router.prefetch(l.href));
  }, [router]);

  useEffect(() => setOpen(false), [pathname]);

  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full transition",
        scrolled
          ? "backdrop-blur supports-[backdrop-filter]:bg-white/60 bg-white/80 ring-1 ring-black/5"
          : "bg-transparent"
      )}
    >
      <nav className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="inline-block h-8 w-8 rounded-xl bg-[color:var(--aerux-navy)]" />
          <span className="text-lg font-semibold text-[color:var(--aerux-navy)]">AERUX</span>
        </Link>

        {/* Desktop */}
        <ul className="hidden items-center gap-6 md:flex">
          {LINKS.map((item) => {
            const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <li key={item.href} className="relative">
                <Link
                  href={item.href}
                  prefetch
                  onMouseEnter={() => router.prefetch(item.href)}
                  className="group inline-flex items-center text-sm font-medium text-[color:var(--aerux-navy)]/85 hover:text-[color:var(--aerux-navy)]"
                >
                  {item.label}
                </Link>
                <span
                  className={cn(
                    "pointer-events-none absolute -bottom-1 left-0 h-[2px] w-full origin-left rounded-full bg-[color:var(--aerux-navy)] transition-transform duration-300",
                    isActive ? "scale-x-100" : "scale-x-0 group-hover:scale-x-100"
                  )}
                />
              </li>
            );
          })}
        </ul>

        {/* Mobile toggle */}
        <button
          onClick={() => setOpen((v) => !v)}
          className="inline-flex h-10 w-10 items-center justify-center rounded-md border md:hidden"
          aria-label="Toggle menu"
        >
          <span className="block h-0.5 w-5 bg-[color:var(--aerux-navy)]" />
          <span className="block h-0.5 w-5 bg-[color:var(--aerux-navy)]" style={{ marginTop: 5 }} />
          <span className="block h-0.5 w-5 bg-[color:var(--aerux-navy)]" style={{ marginTop: 5 }} />
        </button>
      </nav>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden">
          <ul className="mx-4 mb-4 space-y-2 rounded-2xl border bg-white p-3 shadow-sm">
            {LINKS.map((item) => {
              const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
              return (
                <li key={item.href} className="relative">
                  <Link
                    href={item.href}
                    className="block rounded-lg px-3 py-2 text-sm font-medium text-[color:var(--aerux-navy)]/90 hover:bg-zinc-50"
                  >
                    {item.label}
                  </Link>
                  <span
                    className={cn(
                      "pointer-events-none absolute -bottom-0.5 left-3 h-[2px] w-16 origin-left rounded-full bg-[color:var(--aerux-navy)] transition-transform duration-300",
                      isActive ? "scale-x-100" : "scale-x-0"
                    )}
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </header>
  );
}


