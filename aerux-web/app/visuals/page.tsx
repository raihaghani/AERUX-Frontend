"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import { Upload } from "lucide-react";
import { useUIStore } from "@/store/ui";

export default function VisualsPage() {
  const selectedFileName = useUIStore((s) => s.selectedFileName);
  const result = useUIStore((s) => s.predictionResult);
  const modality = useUIStore((s) => s.selectedModality);

  const hasScan = Boolean(selectedFileName);

  return (
    <main className="relative mx-auto w-full max-w-6xl px-6 py-14">
      <div className="mb-6 flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Visual Analytics Dashboard</h1>
          <p className="mt-2 text-zinc-700">Explore results, measurements, and confidence overlays.</p>
        </div>
        {!hasScan && (
          <Link
            href="/upload"
            className="inline-flex h-11 items-center justify-center rounded-2xl bg-[color:var(--aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105"
          >
            Upload Scan
          </Link>
        )}
      </div>

      {!hasScan || !result ? <EmptyState /> : <ResultsSection result={result} modality={modality} />}
    </main>
  );
}

function EmptyState() {
  return (
    <div className="grid place-items-center rounded-2xl bg-white p-16 text-center ring-1 ring-black/10 shadow-sm mt-8">
      <span className="grid h-16 w-16 place-items-center rounded-2xl bg-white ring-1 ring-[color:var(--aerux-navy)] text-[color:var(--aerux-navy)] shadow-sm">
        <Upload className="h-7 w-7" />
      </span>
      <p className="mt-4 text-lg font-medium text-[color:var(--aerux-navy)]">No scan uploaded</p>
      <p className="mt-1 text-sm text-zinc-600">Upload a scan to view analytics and 3D rendering.</p>
      <Link
        href="/upload"
        className="mt-6 inline-flex h-11 items-center justify-center rounded-2xl bg-[color:var(--aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105"
      >
        Go to Upload
      </Link>
    </div>
  );
}

function ResultsSection({ result, modality }: { result: any, modality: string | null }) {
  const isAneurysm = result.detection_prediction === 1;
  
  const stats = [
    { label: "Detection", value: isAneurysm ? "Aneurysm Detected" : "Clear" },
    { label: "Confidence", value: `${(result.detection_probabilities.aneurysm * 100).toFixed(1)}%` },
    { label: "Modality", value: modality || "Unknown" },
    { label: "Top Location", value: result.top_3_locations[0]?.label || "N/A" },
    { label: "Processing Time", value: `${result.processing_time_ms} ms` }
  ];

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.12 } },
  };
  const item = {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } },
  };

  return (
    <div className="space-y-8 mt-8">
      {/* Visual Overlays & 13 Locations Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Visuals Column */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true, amount: 0.2 }}
          transition={fadeUpTransition(0)}
          className="flex flex-col gap-6"
        >
          {/* Main Overlay Image proxy component */}
          <div className="rounded-2xl border border-[color:var(--aerux-blue)]/20 bg-white p-3 shadow-sm flex flex-col items-center">
            <h3 className="text-[color:var(--aerux-navy)] font-semibold text-center mb-2">Algorithm Overlay</h3>
            <img
              src={result.image_urls.overlay}
              alt="Heatmap overlay"
              className="w-full max-h-[400px] rounded-xl object-contain bg-black/5"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-[color:var(--aerux-blue)]/20 bg-white p-2 shadow-sm">
              <p className="text-xs font-medium text-center text-zinc-600 mb-1">Raw Input</p>
              <img src={result.image_urls.input_gray} alt="Input Gray" className="w-full rounded-lg bg-black/5" />
            </div>
            <div className="rounded-xl border border-[color:var(--aerux-blue)]/20 bg-white p-2 shadow-sm">
              <p className="text-xs font-medium text-center text-zinc-600 mb-1">Probability Map</p>
              <img src={result.image_urls.heatmap} alt="Heatmap" className="w-full rounded-lg bg-black/5" />
            </div>
          </div>
        </motion.div>

        {/* Location Probabilities Column */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true, amount: 0.2 }}
          transition={fadeUpTransition(0.1)}
          className="rounded-2xl border border-[color:var(--aerux-blue)]/20 bg-white p-6 shadow-sm flex flex-col h-full"
        >
          <h2 className="text-xl font-bold text-[color:var(--aerux-navy)] mb-6">Location Assessment</h2>
          
          <div className="flex-1 flex flex-col gap-4 overflow-y-auto pr-2 custom-scroll">
            {Object.entries(result.all_location_probabilities)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .map(([label, prob]: [string, any]) => (
                <div key={label} className="flex flex-col gap-1.5 border-b border-zinc-50 pb-3 last:border-0 last:pb-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-zinc-700">{label}</span>
                    <span className="text-xs font-bold text-[color:var(--aerux-navy)]">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-zinc-100 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${prob * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                      className={`h-full rounded-full ${prob > 0.5 ? 'bg-red-500' : 'bg-[color:var(--aerux-accent)]'}`}
                    />
                  </div>
                </div>
              ))}
          </div>
        </motion.div>
      </div>

      {/* Stats cards */}
      <motion.ul
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.2 }}
        className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5"
      >
        {stats.map((s, idx) => (
          <motion.li
            key={idx}
            variants={item}
            className={`group cursor-pointer rounded-2xl bg-white/80 p-5 ring-1 ring-black/5 backdrop-blur supports-[backdrop-filter]:bg-white/60 transition will-change-transform hover:-translate-y-1 hover:shadow-xl transform hover:scale-[1.03] ${s.label === 'Detection' && isAneurysm ? 'border-l-4 border-l-red-500 bg-red-50/50' : 'border-l-4 border-l-[color:var(--aerux-accent)] bg-blue-50/50'}`}
          >
            <div className="block h-full">
              <p className="text-xs text-zinc-600 font-medium uppercase tracking-wider">{s.label}</p>
              <p className={`mt-2 text-lg font-bold truncate ${s.label === 'Detection' && isAneurysm ? 'text-red-700' : 'text-[color:var(--aerux-navy)]'}`}>
                {s.value}
              </p>
            </div>
          </motion.li>
        ))}
      </motion.ul>
    </div>
  );
}
