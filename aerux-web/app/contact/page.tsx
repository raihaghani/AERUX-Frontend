"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";

export default function ContactPage() {
  const cards = [
    {
      title: "Source Code",
      desc: "Browse the repository, issues, and roadmap on GitHub.",
      href: "https://github.com/",
      cta: "Open GitHub",
    },
    {
      title: "Research Paper",
      desc: "Read our methodology, experiments, and results on Overleaf.",
      href: "https://www.overleaf.com/",
      cta: "Open Overleaf",
    },
    {
      title: "Contact Us",
      desc: "Email the team for collaborations and inquiries.",
      href: "mailto:aerux@nuces.edu.pk",
      cta: "Send Email",
    },
  ];

  const container = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.12 } } };
  const item = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } } };

  return (
    <main>
      {/* Dark section with top fade */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white to-transparent" />
        <div className="relative mx-auto w-full max-w-6xl px-6 py-16">
          <motion.div
            initial={fadeUp.initial}
            whileInView={fadeUp.animate}
            viewport={{ once: true, amount: 0.2 }}
            transition={fadeUpTransition(0)}
            className="rounded-3xl bg-[color:var(--aerux-navy)] p-10 text-white shadow"
          >
            <h1 className="text-3xl font-bold tracking-tight">Contact & Resources</h1>
            <p className="mt-2 text-white/80">Access our code, paper, and reach the team.</p>

            <motion.ul
              variants={container}
              initial="hidden"
              whileInView="show"
              viewport={{ once: true, amount: 0.2 }}
              className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
            >
              {cards.map((c) => (
                <motion.li key={c.title} variants={item} className="rounded-2xl bg-white/10 p-6 ring-1 ring-white/15 backdrop-blur">
                  <h3 className="text-lg font-semibold">{c.title}</h3>
                  <p className="mt-2 text-sm text-white/85">{c.desc}</p>
                  <Link
                    href={c.href}
                    target={c.href.startsWith("http") ? "_blank" : undefined}
                    className="mt-4 inline-flex rounded-xl bg-white px-4 py-2 text-sm font-medium text-[color:var(--aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-100 transform transition hover:scale-[1.03]"
                  >
                    {c.cta}
                  </Link>
                </motion.li>
              ))}
            </motion.ul>

            <p className="mt-10 text-center text-xs text-white/70">© 2025–2026 AERUX</p>
          </motion.div>
        </div>
      </section>
    </main>
  );
}



