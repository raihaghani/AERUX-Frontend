"use client";

import { motion } from "framer-motion";
import { Award } from "lucide-react";

export default function AboutPage() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.12 },
    },
  };

  const fadeItem = {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  const principles = [
    { title: "Clinical Accuracy", desc: "Prioritize sensitivity and specificity for reliable diagnostics." },
    { title: "Data Generalizability", desc: "Robust performance across scanners, sites, and populations." },
    { title: "Clinician-in-the-Loop", desc: "Augment radiologists, preserve oversight, and integrate feedback." },
    { title: "Ethical AI", desc: "Privacy, fairness, and transparency at every stage of development." },
  ];

  return (
    <main className="relative mx-auto w-full max-w-6xl px-6 py-16">
      <div className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-bold tracking-tight">About AERUX</h1>
        <p className="mt-4 text-lg text-zinc-700">
          A multi-modal deep learning framework designed to detect and analyze intracranial aneurysms from medical brain scans.
        </p>
      </div>

      {/* Mission card */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" as const }}
        className="mt-10 rounded-2xl bg-white p-6 ring-1 ring-black/10"
      >
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white ring-1 ring-[color:var(--aerux-navy)] text-[color:var(--aerux-navy)] shadow-sm">
            <Award className="h-5 w-5" />
          </span>
          <h2 className="text-xl font-semibold text-[color:var(--aerux-navy)]">Project Mission</h2>
        </div>
        <p className="mt-3 text-zinc-700">
          Deliver clinically useful assistance by accurately localizing and characterizing aneurysms, reducing time-to-diagnosis while maintaining expert oversight.
        </p>
      </motion.div>

      {/* Principles */}
      <section className="mt-12">
        <h3 className="text-lg font-semibold tracking-tight text-[color:var(--aerux-navy)]/90">Our Principles</h3>
        <motion.ul
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.3 }}
          className="mt-6 grid grid-cols-1 gap-4 sm:gap-6 md:grid-cols-2 lg:grid-cols-4"
        >
          {principles.map((p) => (
            <motion.li
              key={p.title}
              variants={fadeItem}
              className="group rounded-2xl bg-white/80 p-5 ring-1 ring-black/5 backdrop-blur supports-[backdrop-filter]:bg-white/60 transition will-change-transform hover:-translate-y-1 hover:shadow-xl"
            >
              <h4 className="text-base font-semibold text-[color:var(--aerux-navy)]">{p.title}</h4>
              <p className="mt-2 text-sm leading-6 text-zinc-700">{p.desc}</p>
            </motion.li>
          ))}
        </motion.ul>
      </section>
    </main>
  );
}





