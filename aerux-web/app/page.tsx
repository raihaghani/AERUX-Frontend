"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import { Brain, AlertTriangle, Target, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

// (moved above to ensure availability at render time)

export default function Home() {
  return (
    <main className="relative w-full">
      <section className="relative flex min-h-[calc(100vh-64px)] items-center justify-center overflow-hidden px-6 py-16">
        {/* Animated network background */}
        <div className="pointer-events-none absolute inset-0">
          <NetworkBackground />
        </div>

        {/* Hero content */}
        <div className="relative z-10 mx-auto flex max-w-3xl flex-col items-center text-center">
          <motion.div
            initial={fadeUp.initial}
            animate={fadeUp.animate}
            transition={fadeUpTransition(0)}
            className="mb-6 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-black/10"
          >
            <Brain className="h-12 w-12 text-[color:var(--aerux-navy)]" />
          </motion.div>

          {/* Behind hero content */}
          <div
            className="absolute top-[-120px] left-1/2 -translate-x-1/2 w-[600px] h-[600px]
                      rounded-full opacity-[0.07] pointer-events-none"
            style={{
              background: "radial-gradient(circle, var(--color-aerux-accent) 0%, transparent 70%)",
              filter: "blur(60px)",
            }}
          />

          <h1
            className="text-5xl md:text-6xl font-bold text-[var(--color-aerux-navy)] leading-[1.1] tracking-tight"
            style={{ fontFamily: "var(--font-display)" }}
          >
            AI-Assisted<br />
            <span className="text-[var(--color-aerux-accent)]">Intracranial Aneurysm Detection</span>
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-zinc-700 relative z-10">
            AI-powered detection of intracranial aneurysms from brain scans
          </p>

          <div className="mt-8 flex w-full flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href="/upload"
              prefetch
              className={cn(
                "inline-flex h-11 items-center justify-center rounded-full px-6 font-semibold text-[14px]",
                "bg-[var(--color-aerux-accent-lt)] text-[var(--color-aerux-navy)] hover:bg-[#DDEBFC] transition-colors shadow-sm"
              )}
            >
              Upload Scan
            </Link>
            <button
              onClick={() => {
                document.getElementById('learn-more-section')?.scrollIntoView({ behavior: 'smooth' });
              }}
              className={cn(
                "inline-flex h-11 items-center justify-center rounded-full px-6 font-medium text-[14px]",
                "border border-[var(--color-border)] text-[var(--color-aerux-navy)] bg-white/70",
                "hover:bg-white transition-colors cursor-pointer"
              )}
            >
              Learn More
            </button>
          </div>
        </div>
      </section>

      {/* Problem, Solution, Impact */}
      <motion.section
        id="learn-more-section"
        initial={fadeUp.initial}
        whileInView={fadeUp.animate}
        viewport={{ once: true, amount: 0.2 }}
        transition={fadeUpTransition(0)}
        className="relative z-10 mx-auto w-full max-w-6xl px-6 pb-20"
      >
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-2xl font-semibold tracking-tight text-[color:var(--aerux-navy)]/90">
            Problem, Solution, Impact
          </h2>
          <p className="mt-2 text-zinc-700">
            How AERUX addresses critical challenges in neurovascular diagnostics.
          </p>
        </div>

        <CardsSection />
      </motion.section>

      {/* How AERUX Works */}
      <HowItWorksSection />
    </main >
  );
}

function HowItWorksSection() {
  const steps = [
    { id: 1, title: "Upload Scan" },
    { id: 2, title: "AI Analysis" },
    { id: 3, title: "Visualizations" },
    { id: 4, title: "Export Report" },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.15 } },
  };

  const item = {
    hidden: { opacity: 0, y: 14 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  return (
    <motion.section
      initial={fadeUp.initial}
      whileInView={fadeUp.animate}
      viewport={{ once: true, amount: 0.2 }}
      transition={fadeUpTransition(0)}
      className="relative z-10 mx-auto w-full max-w-6xl px-6 pb-20"
    >
      <div className="mx-auto max-w-3xl text-center">
        <h2 className="text-2xl font-semibold tracking-tight text-[var(--color-aerux-navy)]/90">
          How AERUX Works
        </h2>
      </div>

      {/* Timeline */}
      <div className="relative mt-8 mt-12 pb-4">
        {/* Base line */}
        <div className="h-[2px] w-full bg-[var(--color-aerux-accent-lt)]" />
        {/* Animated progress line */}
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 1, ease: "easeOut" as const }}
          className="origin-left absolute inset-y-0 left-0 h-[2px] bg-[var(--color-aerux-accent)]"
        />

        {/* Steps */}
        <div className="pointer-events-none absolute -top-6 left-0 right-0 flex items-center justify-between">
          {steps.map((s) => (
            <div key={s.id} className="flex flex-col items-center gap-2">
              <div className="pointer-events-auto grid h-12 w-12 place-items-center rounded-full bg-[var(--color-aerux-navy)] text-white shadow-md transition hover:brightness-110">
                <span className="text-xl font-semibold">{s.id}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Cards under timeline */}
      <motion.ul
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.2 }}
        className="mt-12 grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4"
      >
        {steps.map((s) => (
          <motion.li
            key={s.id}
            variants={item}
            className="group rounded-2xl bg-white/80 p-6 ring-1 ring-black/5 backdrop-blur supports-[backdrop-filter]:bg-white/60 transition will-change-transform hover:-translate-y-1 hover:shadow-xl"
          >
            <h3 className="text-lg font-semibold text-[var(--color-aerux-navy)]">{s.title}</h3>
            <p className="mt-3 text-sm leading-6 text-zinc-700">
              {s.id === 1 && "Securely upload CTA, MRA, or MRI (DICOM / NIfTI). Files are de-identified and prepared for analysis."}
              {s.id === 2 && "AERUX runs multi-modal AI: classification."}
              {s.id === 3 && "Explore results with overlays, heatmaps, and measurements in a clinician-friendly viewer."}
              {s.id === 4 && "One-click export of a structured diagnostic PDF report with findings and notes."}
            </p>
            <div className="mt-4 h-0.5 w-12 origin-left bg-[color:var(--aerux-blue)]/50 transition group-hover:scale-x-110 group-hover:bg-[color:var(--aerux-navy)]" />
          </motion.li>
        ))}
      </motion.ul>
    </motion.section>
  );
}

function NetworkBackground() {
  // Subtle animated network lines using SVG and framer-motion
  const points: Array<[number, number]> = [
    [5, 10],
    [20, 25],
    [40, 15],
    [60, 30],
    [80, 12],
    [15, 70],
    [35, 55],
    [55, 75],
    [75, 60],
    [90, 80],
  ];

  const connections: Array<[number, number]> = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [2, 6],
  ];

  return (
    <svg className="h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(0,116,217,0.35)" />
          <stop offset="100%" stopColor="rgba(0,116,217,0)" />
        </radialGradient>
      </defs>
      {/* Lines */}
      {connections.map(([a, b], idx) => {
        const [x1, y1] = points[a];
        const [x2, y2] = points[b];
        return (
          <motion.line
            key={idx}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="rgba(10,20,72,0.25)"
            strokeWidth={0.2}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.6, 0.2, 0.6] }}
            transition={{ duration: 6 + (idx % 3), repeat: Infinity, ease: "easeInOut" }}
          />
        );
      })}
      {/* Nodes */}
      {points.map(([x, y], idx) => (
        <g key={idx}>
          <circle cx={x} cy={y} r={0.6} fill="rgba(0,116,217,0.6)" />
          <circle cx={x} cy={y} r={3} fill="url(#nodeGlow)" />
        </g>
      ))}
    </svg>
  );
}

function CardsSection() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.15 },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 14 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  const cards = [
    {
      title: "Problem",
      desc: "Aneurysm detection is time-critical and error-prone under workload pressure.",
      icon: AlertTriangle,
    },
    {
      title: "Solution",
      desc: "AERUX augments radiologists with fast, AI-driven aneurysm detection.",
      icon: Target,
    },
    {
      title: "Impact",
      desc: "Improved sensitivity, reduced latency, and streamlined clinical workflows.",
      icon: Zap,
    },
  ];

  return (
    <motion.ul
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.3 }}
      className="mt-8 grid grid-cols-1 gap-4 sm:gap-6 md:grid-cols-3"
    >
      {cards.map(({ title, desc, icon: Icon }) => (
        <motion.li
          key={title}
          variants={item}
          className="group rounded-2xl bg-white/80 p-6 ring-1 ring-black/5 backdrop-blur supports-[backdrop-filter]:bg-white/60 transition will-change-transform hover:-translate-y-1 hover:shadow-xl"
        >
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white ring-1 ring-[color:var(--aerux-navy)] text-[color:var(--aerux-navy)] shadow-sm">
              <Icon className="h-5 w-5" />
            </span>
            <h3 className="text-lg font-semibold text-[color:var(--aerux-navy)]">{title}</h3>
          </div>
          <p className="mt-3 text-sm leading-6 text-zinc-700">{desc}</p>
        </motion.li>
      ))}
    </motion.ul>
  );
}


