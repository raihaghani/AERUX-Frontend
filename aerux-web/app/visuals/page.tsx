"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import { Upload, AlertTriangle, CheckCircle } from "lucide-react";
import { useUIStore, type SeriesResult, type SeriesTopResult, type SeriesSliceResult } from "@/store/ui";

export default function VisualsPage() {
  const selectedFileName = useUIStore((s) => s.selectedFileName);
  const result = useUIStore((s) => s.predictionResult);
  const seriesResult = useUIStore((s) => s.seriesResult);
  const modality = useUIStore((s) => s.selectedModality);

  if (seriesResult) {
    return <SeriesResultsSection result={seriesResult} modality={modality} />;
  }

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
            className="inline-flex h-11 items-center justify-center rounded-2xl bg-[var(--color-aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105"
          >
            Upload Scan
          </Link>
        )}
      </div>

      {!hasScan || !result ? <EmptyState /> : <ResultsSection result={result} modality={modality} />}
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Empty state
   ═══════════════════════════════════════════════════════════════════════════════ */

function EmptyState() {
  return (
    <div className="grid place-items-center rounded-2xl bg-white p-16 text-center ring-1 ring-black/10 shadow-sm mt-8">
      <span className="grid h-16 w-16 place-items-center rounded-2xl bg-white ring-1 ring-[var(--color-aerux-navy)] text-[var(--color-aerux-navy)] shadow-sm">
        <Upload className="h-7 w-7" />
      </span>
      <p className="mt-4 text-lg font-medium text-[var(--color-aerux-navy)]">No scan uploaded</p>
      <p className="mt-1 text-sm text-zinc-600">Upload a scan to view analytics and 3D rendering.</p>
      <Link
        href="/upload"
        className="mt-6 inline-flex h-11 items-center justify-center rounded-2xl bg-[var(--color-aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105"
      >
        Go to Upload
      </Link>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Single-image results (unchanged from original)
   ═══════════════════════════════════════════════════════════════════════════════ */

function ResultsSection({ result, modality }: { result: any; modality: string | null }) {
  const isAneurysm = result.detection_prediction === 1;

  const stats = [
    { label: "Detection", value: isAneurysm ? "Aneurysm Detected" : "Clear" },
    { label: "Confidence", value: `${(result.detection_probabilities.aneurysm * 100).toFixed(1)}%` },
    { label: "Modality", value: modality || "Unknown" },
    { label: "Top Location", value: result.top_3_locations[0]?.label || "N/A" },
    { label: "Processing Time", value: `${result.processing_time_ms} ms` },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.12 } },
  };
  const item = {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" as const } },
  };

  return (
    <div className="space-y-8 mt-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Visuals Column */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true, amount: 0.2 }}
          transition={fadeUpTransition(0)}
          className="flex flex-col gap-6"
        >
          <div className="rounded-2xl border border-[var(--color-aerux-blue)]/20 bg-white p-3 shadow-sm flex flex-col items-center">
            <h3 className="text-[var(--color-aerux-navy)] font-semibold text-center mb-2">Algorithm Overlay</h3>
            <img
              src={result.image_urls.overlay}
              alt="Heatmap overlay"
              className="w-full max-h-[400px] rounded-xl object-contain bg-black/5"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-[var(--color-aerux-blue)]/20 bg-white p-2 shadow-sm">
              <p className="text-xs font-medium text-center text-zinc-600 mb-1">Raw Input</p>
              <img src={result.image_urls.input_gray} alt="Input Gray" className="w-full rounded-lg bg-black/5" />
            </div>
            <div className="rounded-xl border border-[var(--color-aerux-blue)]/20 bg-white p-2 shadow-sm">
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
          className="rounded-2xl border border-[var(--color-aerux-blue)]/20 bg-white p-6 shadow-sm flex flex-col h-full"
        >
          <h2 className="text-xl font-bold text-[var(--color-aerux-navy)] mb-6">Location Assessment</h2>

          <div className="flex-1 flex flex-col gap-4 overflow-y-auto pr-2 custom-scroll">
            {Object.entries(result.all_location_probabilities)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .slice(0, 3)
              .map(([label, prob]: [string, any]) => (
                <div key={label} className="flex flex-col gap-1.5 border-b border-zinc-50 pb-3 last:border-0 last:pb-0">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-zinc-700">{label}</span>
                    <span className="text-xs font-bold text-[var(--color-aerux-navy)]">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-zinc-100 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${prob * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" as const, delay: 0.2 }}
                      className={`h-full rounded-full ${prob > 0.5 ? "bg-red-500" : "bg-[var(--color-aerux-accent)]"}`}
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
            className={`group cursor-pointer rounded-2xl bg-white/80 p-5 ring-1 ring-black/5 backdrop-blur supports-[backdrop-filter]:bg-white/60 transition will-change-transform hover:-translate-y-1 hover:shadow-xl transform hover:scale-[1.03] ${
              s.label === "Detection" && isAneurysm
                ? "border-l-4 border-l-red-500 bg-red-50/50"
                : "border-l-4 border-l-[var(--color-aerux-accent)] bg-blue-50/50"
            }`}
          >
            <div className="block h-full">
              <p className="text-xs text-zinc-600 font-medium uppercase tracking-wider">{s.label}</p>
              <p
                className={`mt-2 text-lg font-bold truncate ${
                  s.label === "Detection" && isAneurysm ? "text-red-700" : "text-[var(--color-aerux-navy)]"
                }`}
              >
                {s.value}
              </p>
            </div>
          </motion.li>
        ))}
      </motion.ul>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Series scan results
   ═══════════════════════════════════════════════════════════════════════════════ */

function SeriesResultsSection({
  result,
  modality,
}: {
  result: SeriesResult;
  modality: string | null;
}) {
  const selectedSlice = useUIStore((s) => s.selectedSeriesSlice);
  const setSelectedSlice = useUIStore((s) => s.setSelectedSeriesSlice);
  const dynamicSlices = useUIStore((s) => s.dynamicSlices);
  const setDynamicSlices = useUIStore((s) => s.setDynamicSlices);
  const [loadingSlice, setLoadingSlice] = useState(false);
  
  const anyFlagged = result.flagged_windows > 0;

  const stats = [
    { label: "Total Slices", value: String(result.total_slices) },
    { label: "Windows Scanned", value: String(result.total_windows) },
    { label: "Max Probability", value: `${(result.max_aneurysm_prob * 100).toFixed(1)}%` },
    { label: "Flagged Windows", value: String(result.flagged_windows) },
    { label: "Modality", value: modality || "Unknown" },
    { label: "Processing Time", value: `${(result.processing_time_ms / 1000).toFixed(1)}s` },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.08 } },
  };
  const item = {
    hidden: { opacity: 0, y: 12 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
  };

  const handleSelectSlice = async (slice: number) => {
    if (slice === selectedSlice) return;
    setSelectedSlice(slice);

    const isInTopResults = result.top_results.some((tr) => tr.center_slice === slice);
    if (isInTopResults || dynamicSlices[slice]) {
      return;
    }

    setLoadingSlice(true);
    try {
      const res = await fetch(`/api/slice-details?result_id=${result.result_id}&center_slice=${slice}`);
      if (!res.ok) {
        throw new Error("Failed to fetch slice details");
      }
      const data = await res.json();
      setDynamicSlices((prev) => ({ ...prev, [slice]: data }));
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSlice(false);
    }
  };

  const selectedData =
    result.top_results.find((tr) => tr.center_slice === selectedSlice) ||
    (selectedSlice !== null ? dynamicSlices[selectedSlice] : null);

  return (
    <main className="relative mx-auto w-full max-w-6xl px-6 py-14">
      {/* Header */}
      <div className="mb-6 flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Series Scan Results</h1>
          <p className="mt-2 text-zinc-700">
            Sliding-window analysis across {result.total_slices} slices (step&nbsp;{result.step}).
          </p>
        </div>
        <Link
          href="/upload"
          className="inline-flex h-11 items-center justify-center rounded-2xl bg-[var(--color-aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105"
        >
          New Scan
        </Link>
      </div>

      {/* Summary banner */}
      <motion.div
        initial={fadeUp.initial}
        whileInView={fadeUp.animate}
        viewport={{ once: true, amount: 0.2 }}
        transition={fadeUpTransition(0)}
        className={`rounded-2xl p-5 shadow ${
          anyFlagged
            ? "bg-gradient-to-r from-red-600 to-red-500 text-white"
            : "bg-gradient-to-r from-emerald-600 to-emerald-500 text-white"
        }`}
      >
        <div className="flex items-center gap-3">
          {anyFlagged ? <AlertTriangle className="h-6 w-6" /> : <CheckCircle className="h-6 w-6" />}
          <span className="text-lg font-semibold">
            {anyFlagged
              ? `Potential aneurysm detected in ${result.flagged_windows} window${result.flagged_windows > 1 ? "s" : ""}`
              : "No aneurysm detected across all windows"}
          </span>
        </div>
      </motion.div>

      {/* Stats cards */}
      <motion.ul
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.2 }}
        className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6"
      >
        {stats.map((s, idx) => (
          <motion.li
            key={idx}
            variants={item}
            className={`rounded-2xl bg-white/80 p-4 ring-1 ring-black/5 backdrop-blur ${
              s.label === "Flagged Windows" && anyFlagged
                ? "border-l-4 border-l-red-500 bg-red-50/50"
                : "border-l-4 border-l-[var(--color-aerux-accent)] bg-blue-50/50"
            }`}
          >
            <p className="text-xs text-zinc-600 font-medium uppercase tracking-wider">{s.label}</p>
            <p className="mt-1 text-lg font-bold text-[var(--color-aerux-navy)]">{s.value}</p>
          </motion.li>
        ))}
      </motion.ul>

      {/* Scan profile */}
      <ScanProfile 
        sliceResults={result.slice_results} 
        selectedSlice={selectedSlice}
        onSliceSelect={handleSelectSlice}
      />

      {/* Top suspicious windows */}
      {result.top_results.length > 0 && (
        <div className="mt-8 space-y-6">
          <h2 className="text-xl font-bold text-[var(--color-aerux-navy)]">
            Top Suspicious Windows
          </h2>

          {/* Tab bar */}
          <div className="flex gap-2 overflow-x-auto pb-1">
            {result.top_results.map((tr, i) => {
              const isAneurysm = tr.detection_prediction === 1;
              return (
                <button
                  key={i}
                  onClick={() => handleSelectSlice(tr.center_slice)}
                  className={`flex-shrink-0 rounded-xl px-4 py-2 text-sm font-medium transition ring-1 ${
                    selectedSlice === tr.center_slice
                      ? "bg-[var(--color-aerux-navy)] text-white ring-[var(--color-aerux-navy)]"
                      : isAneurysm
                        ? "bg-red-50 text-red-700 ring-red-200 hover:bg-red-100"
                        : "bg-zinc-50 text-zinc-700 ring-zinc-200 hover:bg-zinc-100"
                  }`}
                >
                  Slice {tr.center_slice}{" "}
                  <span className="ml-1 text-xs opacity-75">
                    {(tr.detection_probabilities.aneurysm * 100).toFixed(1)}%
                  </span>
                </button>
              );
            })}
          </div>

          {/* Detail view for selected window */}
          {loadingSlice ? (
            <div className="flex justify-center items-center py-12">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-[var(--color-aerux-navy)] border-t-transparent" />
              <span className="ml-3 text-zinc-600 font-medium">Generating AI overlays for slice {selectedSlice}...</span>
            </div>
          ) : selectedData ? (
            <WindowDetail result={selectedData} />
          ) : null}
        </div>
      )}
    </main>
  );
}

/* ── Scan profile chart ────────────────────────────────────────────────────── */

function ScanProfile({ 
  sliceResults,
  selectedSlice,
  onSliceSelect
}: { 
  sliceResults: SeriesSliceResult[];
  selectedSlice: number | null;
  onSliceSelect: (slice: number) => void;
}) {
  if (sliceResults.length === 0) return null;

  const chartHeight = 100;

  return (
    <motion.div
      initial={fadeUp.initial}
      whileInView={fadeUp.animate}
      viewport={{ once: true, amount: 0.2 }}
      transition={fadeUpTransition(0.15)}
      className="mt-8 rounded-2xl border border-[var(--color-aerux-blue)]/20 bg-white p-6 shadow-sm"
    >
      <h2 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-4">
        Scan Profile — Aneurysm Probability per Window
      </h2>

      <div className="relative overflow-x-auto">
        <div
          className="flex items-end gap-[1px]"
          style={{ minHeight: `${chartHeight + 16}px` }}
        >
          {sliceResults.map((r, i) => {
            const h = Math.max(3, r.aneurysm_prob * chartHeight);
            const bg =
              r.aneurysm_prob >= 0.5
                ? "bg-red-500"
                : r.aneurysm_prob >= 0.3
                  ? "bg-amber-400"
                  : "bg-emerald-400";
            const barWidth = Math.max(3, Math.min(10, Math.floor(700 / sliceResults.length)));
            return (
              <div
                key={i}
                title={`Slice ${r.center_slice}: ${(r.aneurysm_prob * 100).toFixed(1)}%`}
                style={{ width: `${barWidth}px`, height: `${h}px` }}
                onClick={() => onSliceSelect(r.center_slice)}
                className={`rounded-t-sm ${bg} flex-shrink-0 cursor-pointer transition-all hover:opacity-100 ${
                  r.center_slice === selectedSlice ? "ring-2 ring-zinc-800 ring-offset-1 z-10 opacity-100 scale-y-105" : "opacity-80"
                }`}
              />
            );
          })}
        </div>

        {/* 50% threshold line */}
        <div
          className="absolute left-0 right-0 border-t-2 border-dashed border-red-300 pointer-events-none"
          style={{ bottom: `${chartHeight * 0.5 + 16}px` }}
        />
      </div>

      <div className="flex justify-between text-xs text-zinc-500 mt-3">
        <span>Slice {sliceResults[0]?.center_slice}</span>
        <span className="text-red-400 font-medium">50% threshold</span>
        <span>Slice {sliceResults[sliceResults.length - 1]?.center_slice}</span>
      </div>
    </motion.div>
  );
}

/* ── Detail view for a single window ───────────────────────────────────────── */

function WindowDetail({ result }: { result: SeriesTopResult }) {
  const isAneurysm = result.detection_prediction === 1;

  return (
    <motion.div
      key={result.center_slice}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="grid grid-cols-1 lg:grid-cols-2 gap-6"
    >
      {/* Images */}
      <div className="flex flex-col gap-4">
        <div className="rounded-2xl border border-[var(--color-aerux-blue)]/20 bg-white p-3 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[var(--color-aerux-navy)] font-semibold">
              Overlay — Slice {result.center_slice}
            </h3>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                isAneurysm ? "bg-red-100 text-red-700" : "bg-emerald-100 text-emerald-700"
              }`}
            >
              {isAneurysm ? "Aneurysm" : "Clear"}
            </span>
          </div>
          <img
            src={result.image_urls.overlay}
            alt={`Overlay slice ${result.center_slice}`}
            className="w-full max-h-[350px] rounded-xl object-contain bg-black/5"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-[var(--color-aerux-blue)]/20 bg-white p-2 shadow-sm">
            <p className="text-xs font-medium text-center text-zinc-600 mb-1">Raw Input</p>
            <img src={result.image_urls.input_gray} alt="Input" className="w-full rounded-lg bg-black/5" />
          </div>
          <div className="rounded-xl border border-[var(--color-aerux-blue)]/20 bg-white p-2 shadow-sm">
            <p className="text-xs font-medium text-center text-zinc-600 mb-1">Probability Map</p>
            <img src={result.image_urls.heatmap} alt="Heatmap" className="w-full rounded-lg bg-black/5" />
          </div>
        </div>
      </div>

      {/* Location assessment */}
      <div className="rounded-2xl border border-[var(--color-aerux-blue)]/20 bg-white p-6 shadow-sm flex flex-col">
        <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-1">Location Assessment</h3>
        <p className="text-sm text-zinc-500 mb-4">
          Confidence: {(result.detection_probabilities.aneurysm * 100).toFixed(1)}%
        </p>

        <div className="flex-1 flex flex-col gap-3 overflow-y-auto pr-2">
          {Object.entries(result.all_location_probabilities)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 3)
            .map(([label, prob]) => (
              <div key={label} className="flex flex-col gap-1.5 border-b border-zinc-50 pb-2 last:border-0 last:pb-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-zinc-700">{label}</span>
                  <span className="text-xs font-bold text-[var(--color-aerux-navy)]">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-zinc-100 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${prob * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" as const, delay: 0.1 }}
                    className={`h-full rounded-full ${prob > 0.5 ? "bg-red-500" : "bg-[var(--color-aerux-accent)]"}`}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>
    </motion.div>
  );
}



