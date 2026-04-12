"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import {
  CheckCircle2,
  Download,
  AlertCircle,
  Printer,
  FileText,
  Activity,
  MapPin,
  Clock,
  Shield,
  Upload,
} from "lucide-react";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";
import { useUIStore, type SeriesResult, type SeriesTopResult } from "@/store/ui";

/* ═══════════════════════════════════════════════════════════════════════════════
   Main page
   ═══════════════════════════════════════════════════════════════════════════════ */

export default function ReportsPage() {
  const result = useUIStore((s) => s.predictionResult);
  const seriesResult = useUIStore((s) => s.seriesResult);
  const modality = useUIStore((s) => s.selectedModality);
  const fileName = useUIStore((s) => s.selectedFileName);

  const patientName = useUIStore((s) => s.patientName);
  const patientAge = useUIStore((s) => s.patientAge);
  const patientGender = useUIStore((s) => s.patientGender);

  if (seriesResult) {
    return <SeriesReport result={seriesResult} modality={modality} fileName={fileName} patient={{ patientName, patientAge, patientGender }} />;
  }
  if (result) {
    return <SingleReport result={result} modality={modality} fileName={fileName} patient={{ patientName, patientAge, patientGender }} />;
  }
  return <EmptyReport />;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Empty state
   ═══════════════════════════════════════════════════════════════════════════════ */

function EmptyReport() {
  return (
    <main className="relative mx-auto w-full max-w-5xl px-6 py-14">
      <h1 className="text-3xl font-bold tracking-tight">Diagnostic Report</h1>
      <div className="mt-8 grid place-items-center rounded-2xl bg-white p-16 text-center ring-1 ring-black/10 shadow-sm">
        <span className="grid h-16 w-16 place-items-center rounded-2xl bg-white ring-1 ring-[var(--color-aerux-navy)] text-[var(--color-aerux-navy)] shadow-sm">
          <Upload className="h-7 w-7" />
        </span>
        <p className="mt-4 text-lg font-medium text-[var(--color-aerux-navy)]">
          No report available
        </p>
        <p className="mt-1 text-sm text-zinc-600">
          Upload and analyze a scan to generate a diagnostic report.
        </p>
        <Link
          href="/upload"
          className="mt-6 inline-flex h-11 items-center justify-center rounded-2xl bg-[var(--color-aerux-accent)] px-5 font-medium text-white shadow transition hover:brightness-105 transform hover:scale-[1.03]"
        >
          Go to Upload
        </Link>
      </div>
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Single-image report
   ═══════════════════════════════════════════════════════════════════════════════ */

function SingleReport({
  result,
  modality,
  fileName,
  patient,
}: {
  result: any;
  modality: string | null;
  fileName: string | null;
  patient: { patientName: string; patientAge: string; patientGender: string };
}) {
  const isAneurysm = result.detection_prediction === 1;
  const prob = result.detection_probabilities.aneurysm;
  const todayDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const locations = Object.entries(result.all_location_probabilities)
    .sort(([, a], [, b]) => (b as number) - (a as number)) as [string, number][];

  const [downloading, setDownloading] = useState(false);

  return (
    <main className="relative mx-auto w-full max-w-5xl px-6 py-14 print:px-0 print:py-0">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-8 print:hidden">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Diagnostic Report</h1>
          <p className="mt-1 text-zinc-600">
            AI-generated aneurysm detection analysis
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-medium text-[var(--color-aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-50 transition"
          >
            <Printer className="h-4 w-4" /> Print
          </button>
          <button
            onClick={() => downloadSinglePdf(result, modality, todayDate, fileName, patient, setDownloading)}
            disabled={downloading}
            className="inline-flex items-center gap-2 rounded-2xl bg-[var(--color-aerux-navy)] px-4 py-2 text-sm font-medium text-white shadow transition hover:brightness-110 disabled:opacity-60"
          >
            <Download className="h-4 w-4" />
            {downloading ? "Generating..." : "Download PDF"}
          </button>
        </div>
      </div>

      {/* Report body */}
      <div className="space-y-6">
        {/* Status banner */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0)}
          className={`rounded-2xl p-6 shadow ${isAneurysm
            ? "bg-gradient-to-r from-red-600 to-red-500"
            : "bg-gradient-to-r from-emerald-600 to-emerald-500"
            } text-white`}
        >
          <div className="flex items-center gap-3">
            {isAneurysm ? (
              <AlertCircle className="h-7 w-7 shrink-0" />
            ) : (
              <CheckCircle2 className="h-7 w-7 shrink-0" />
            )}
            <div>
              <h2 className="text-xl font-bold">
                {isAneurysm ? "Aneurysm Detected" : "No Aneurysm Detected"}
              </h2>
              <p className="mt-0.5 text-sm text-white/80">
                Detection confidence: {(prob * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </motion.div>

        {/* Patient & scan metadata */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.05)}
          className="grid grid-cols-2 gap-4 sm:grid-cols-4"
        >
          {[
            { icon: FileText, label: "Patient", value: patient.patientName || `P-${result.result_id?.toUpperCase().slice(0, 5) || "00000"}` },
            { icon: Activity, label: "Age / Sex", value: `${patient.patientAge || "U"} / ${patient.patientGender ? patient.patientGender.charAt(0) : "U"}` },
            { icon: Clock, label: "Report Date", value: todayDate },
            { icon: Activity, label: "Modality", value: modality || "Auto-detected" },
          ].map((m, i) => (
            <div
              key={i}
              className="rounded-xl bg-white p-4 ring-1 ring-black/5 shadow-sm"
            >
              <div className="flex items-center gap-2 text-zinc-500 mb-1">
                <m.icon className="h-3.5 w-3.5" />
                <span className="text-xs font-medium uppercase tracking-wider">{m.label}</span>
              </div>
              <p className="text-sm font-bold text-[var(--color-aerux-navy)] truncate">
                {m.value}
              </p>
            </div>
          ))}
        </motion.div>

        {/* Scan images */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.1)}
          className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
        >
          <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-4 flex items-center gap-2">
            <Shield className="h-5 w-5" /> Scan Imaging
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ImageCard
              label="Original Input"
              sublabel="Raw scan image as received"
              src={result.image_urls.input_gray}
            />
            <ImageCard
              label="AI Overlay"
              sublabel="Highlighted regions of interest"
              src={result.image_urls.overlay}
            />
            <ImageCard
              label="Probability Heatmap"
              sublabel="Location-based probability distribution"
              src={result.image_urls.heatmap}
            />
            <ImageCard
              label="Detection Mask"
              sublabel="Binary threshold highlight"
              src={result.image_urls.highlight_mask}
            />
          </div>
        </motion.div>

        {/* Location assessment */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.15)}
          className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
        >
          <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-1 flex items-center gap-2">
            <MapPin className="h-5 w-5" /> Location Assessment
          </h3>
          <p className="text-sm text-zinc-500 mb-5">
            Probability of aneurysm at each anatomical location
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
            {locations.map(([label, prob]) => (
              <div key={label} className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-zinc-700">{label}</span>
                  <span
                    className={`text-xs font-bold ${prob > 0.5 ? "text-red-600" : "text-[var(--color-aerux-navy)]"
                      }`}
                  >
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-zinc-100 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${prob * 100}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8, ease: "easeOut" as const }}
                    className={`h-full rounded-full ${prob > 0.5 ? "bg-red-500" : "bg-[var(--color-aerux-accent)]"
                      }`}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Clinical summary */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.2)}
          className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
        >
          <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-3">
            Clinical Summary
          </h3>
          <div className="text-sm text-zinc-700 leading-relaxed space-y-2">
            <p>
              {isAneurysm
                ? `AI analysis detected a potential intracranial aneurysm with ${(prob * 100).toFixed(1)}% confidence. The most probable location is the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] ?? 0) * 100).toFixed(1)}% probability), followed by ${locations[1]?.[0] || "N/A"} (${((locations[1]?.[1] ?? 0) * 100).toFixed(1)}%) and ${locations[2]?.[0] || "N/A"} (${((locations[2]?.[1] ?? 0) * 100).toFixed(1)}%).`
                : `AI analysis did not detect an intracranial aneurysm. The overall detection confidence for aneurysm was ${(prob * 100).toFixed(1)}%. The highest location probability was at the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] ?? 0) * 100).toFixed(1)}%).`}
            </p>
            <p>
              Scan file: <span className="font-medium">{fileName || "Unknown"}</span> ·
              Modality: <span className="font-medium">{modality || "Auto-detected"}</span> ·
              Processed in <span className="font-medium">{result.processing_time_ms} ms</span>.
            </p>
          </div>
        </motion.div>

        {/* Disclaimer */}
        <Disclaimer />
      </div>
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Series report
   ═══════════════════════════════════════════════════════════════════════════════ */

function SeriesReport({
  result,
  modality,
  fileName,
  patient,
}: {
  result: SeriesResult;
  modality: string | null;
  fileName: string | null;
  patient: { patientName: string; patientAge: string; patientGender: string };
}) {
  const anyFlagged = result.flagged_windows > 0;
  const topWindow = result.top_results[0];
  const todayDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const [downloading, setDownloading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const selected = result.top_results[selectedIdx];

  return (
    <main className="relative mx-auto w-full max-w-5xl px-6 py-14 print:px-0 print:py-0">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-8 print:hidden">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Series Scan Report</h1>
          <p className="mt-1 text-zinc-600">
            Sliding-window analysis across {result.total_slices} slices
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-medium text-[var(--color-aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-50 transition"
          >
            <Printer className="h-4 w-4" /> Print
          </button>
          <button
            onClick={() =>
              downloadSeriesPdf(result, modality, todayDate, fileName, patient, setDownloading)
            }
            disabled={downloading}
            className="inline-flex items-center gap-2 rounded-2xl bg-[var(--color-aerux-navy)] px-4 py-2 text-sm font-medium text-white shadow transition hover:brightness-110 disabled:opacity-60"
          >
            <Download className="h-4 w-4" />
            {downloading ? "Generating..." : "Download PDF"}
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Status banner */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0)}
          className={`rounded-2xl p-6 shadow ${anyFlagged
            ? "bg-gradient-to-r from-red-600 to-red-500"
            : "bg-gradient-to-r from-emerald-600 to-emerald-500"
            } text-white`}
        >
          <div className="flex items-center gap-3">
            {anyFlagged ? <AlertCircle className="h-7 w-7 shrink-0" /> : <CheckCircle2 className="h-7 w-7 shrink-0" />}
            <div>
              <h2 className="text-xl font-bold">
                {anyFlagged
                  ? `Potential aneurysm in ${result.flagged_windows} window${result.flagged_windows > 1 ? "s" : ""}`
                  : "No aneurysm detected across all windows"}
              </h2>
              <p className="mt-0.5 text-sm text-white/80">
                Peak probability: {(result.max_aneurysm_prob * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </motion.div>

        {/* Metadata cards */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.05)}
          className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6"
        >
          {[
            { label: "Patient", value: patient.patientName || `P-${result.result_id?.toUpperCase().slice(0, 5) || "000"}` },
            { label: "Age / Sex", value: `${patient.patientAge || "U"} / ${patient.patientGender ? patient.patientGender.charAt(0) : "U"}` },
            { label: "Date", value: todayDate.split(",")[0] },
            { label: "Slices", value: String(result.total_slices) },
            { label: "Flagged", value: String(result.flagged_windows) },
            { label: "Time", value: `${(result.processing_time_ms / 1000).toFixed(1)}s` },
          ].map((m, i) => (
            <div
              key={i}
              className={`rounded-xl bg-white p-4 ring-1 ring-black/5 shadow-sm ${m.label === "Flagged" && anyFlagged ? "border-l-4 border-l-red-500" : ""
                }`}
            >
              <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">{m.label}</p>
              <p className="mt-1 text-sm font-bold text-[var(--color-aerux-navy)]">{m.value}</p>
            </div>
          ))}
        </motion.div>

        {/* Top windows tab selector + images */}
        {result.top_results.length > 0 && (
          <motion.div
            initial={fadeUp.initial}
            whileInView={fadeUp.animate}
            viewport={{ once: true }}
            transition={fadeUpTransition(0.1)}
            className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
          >
            <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5" /> Top Suspicious Windows
            </h3>

            {/* Tabs */}
            <div className="flex gap-2 overflow-x-auto pb-3 mb-4">
              {result.top_results.map((tr, i) => {
                const det = tr.detection_prediction === 1;
                return (
                  <button
                    key={i}
                    onClick={() => setSelectedIdx(i)}
                    className={`flex-shrink-0 rounded-lg px-3 py-1.5 text-sm font-medium transition ring-1 ${selectedIdx === i
                      ? "bg-[var(--color-aerux-navy)] text-white ring-[var(--color-aerux-navy)]"
                      : det
                        ? "bg-red-50 text-red-700 ring-red-200 hover:bg-red-100"
                        : "bg-zinc-50 text-zinc-700 ring-zinc-200 hover:bg-zinc-100"
                      }`}
                  >
                    Slice {tr.center_slice}
                    <span className="ml-1 text-xs opacity-70">
                      ({(tr.detection_probabilities.aneurysm * 100).toFixed(0)}%)
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Images for selected window */}
            {selected && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ImageCard
                  label="Original Input"
                  sublabel={`Center slice ${selected.center_slice}`}
                  src={selected.image_urls.input_gray}
                />
                <ImageCard
                  label="AI Overlay"
                  sublabel="Highlighted regions of interest"
                  src={selected.image_urls.overlay}
                />
                <ImageCard
                  label="Probability Heatmap"
                  sublabel="Location-based probability"
                  src={selected.image_urls.heatmap}
                />
                <ImageCard
                  label="Detection Mask"
                  sublabel="Binary threshold highlight"
                  src={selected.image_urls.highlight_mask}
                />
              </div>
            )}
          </motion.div>
        )}

        {/* Location assessment for top window */}
        {topWindow && (
          <motion.div
            initial={fadeUp.initial}
            whileInView={fadeUp.animate}
            viewport={{ once: true }}
            transition={fadeUpTransition(0.15)}
            className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
          >
            <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-1 flex items-center gap-2">
              <MapPin className="h-5 w-5" /> Location Assessment
              <span className="text-sm font-normal text-zinc-500">
                (Top window — Slice {topWindow.center_slice})
              </span>
            </h3>
            <p className="text-sm text-zinc-500 mb-5">
              Probability of aneurysm at each anatomical location
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
              {(
                Object.entries(topWindow.all_location_probabilities).sort(
                  ([, a], [, b]) => b - a,
                ) as [string, number][]
              ).map(([label, p]) => (
                <div key={label} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-zinc-700">{label}</span>
                    <span
                      className={`text-xs font-bold ${p > 0.5 ? "text-red-600" : "text-[var(--color-aerux-navy)]"
                        }`}
                    >
                      {(p * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-zinc-100 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${p * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, ease: "easeOut" as const }}
                      className={`h-full rounded-full ${p > 0.5 ? "bg-red-500" : "bg-[var(--color-aerux-accent)]"
                        }`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Clinical summary */}
        <motion.div
          initial={fadeUp.initial}
          whileInView={fadeUp.animate}
          viewport={{ once: true }}
          transition={fadeUpTransition(0.2)}
          className="rounded-2xl bg-white p-6 ring-1 ring-black/5 shadow-sm"
        >
          <h3 className="text-lg font-bold text-[var(--color-aerux-navy)] mb-3">
            Clinical Summary
          </h3>
          <div className="text-sm text-zinc-700 leading-relaxed space-y-2">
            <p>
              {anyFlagged
                ? `Series scan analyzed ${result.total_slices} slices with a sliding window (step ${result.step}). ${result.flagged_windows} of ${result.total_windows} windows flagged potential aneurysm (probability >= 50%). The highest confidence was ${(result.max_aneurysm_prob * 100).toFixed(1)}% at slice ${topWindow?.center_slice ?? "N/A"}.`
                : `Series scan analyzed ${result.total_slices} slices with a sliding window (step ${result.step}). No windows exceeded the 50% aneurysm probability threshold. The highest observed probability was ${(result.max_aneurysm_prob * 100).toFixed(1)}%.`}
            </p>
            {topWindow && (
              <p>
                Top location for the most suspicious window:{" "}
                <span className="font-medium">{topWindow.top_3_locations[0]?.label || "N/A"}</span>{" "}
                ({((topWindow.top_3_locations[0]?.score ?? 0) * 100).toFixed(1)}%).
              </p>
            )}
            <p>
              File: <span className="font-medium">{fileName || "Unknown"}</span> ·
              Modality: <span className="font-medium">{modality || "Auto-detected"}</span> ·
              Processed in <span className="font-medium">{(result.processing_time_ms / 1000).toFixed(1)}s</span>.
            </p>
          </div>
        </motion.div>

        <Disclaimer />
      </div>
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Shared components
   ═══════════════════════════════════════════════════════════════════════════════ */

function ImageCard({ label, sublabel, src }: { label: string; sublabel: string; src: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
      <p className="text-sm font-semibold text-[var(--color-aerux-navy)]">{label}</p>
      <p className="text-xs text-zinc-500 mb-2">{sublabel}</p>
      <img
        src={src}
        alt={label}
        className="w-full rounded-lg bg-black/5 object-contain max-h-[280px]"
      />
    </div>
  );
}

function Disclaimer() {
  return (
    <div className="rounded-xl bg-amber-50 p-4 ring-1 ring-amber-200 text-xs text-amber-800 leading-relaxed">
      <strong>Disclaimer:</strong> This report is generated by an AI system and is
      intended for research and screening purposes only. It does not constitute a
      medical diagnosis. All findings should be reviewed and confirmed by a qualified
      radiologist or neurosurgeon before any clinical decisions are made. AERUX AI
      accepts no liability for clinical decisions based solely on this report.
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   PDF generation helpers
   ═══════════════════════════════════════════════════════════════════════════════ */

async function fetchImageAsBytes(url: string): Promise<Uint8Array | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return new Uint8Array(await res.arrayBuffer());
  } catch {
    return null;
  }
}

async function embedImage(
  pdfDoc: PDFDocument,
  bytes: Uint8Array | null,
): Promise<Awaited<ReturnType<typeof pdfDoc.embedPng>> | null> {
  if (!bytes) return null;
  try {
    return await pdfDoc.embedPng(bytes);
  } catch {
    try {
      return await pdfDoc.embedJpg(bytes);
    } catch {
      return null;
    }
  }
}

function drawSectionTitle(
  page: ReturnType<PDFDocument["addPage"]>,
  font: Awaited<ReturnType<typeof PDFDocument.prototype.embedFont>>,
  boldFont: Awaited<ReturnType<typeof PDFDocument.prototype.embedFont>>,
  text: string,
  y: number,
) {
  page.drawRectangle({ x: 50, y: y - 4, width: 512, height: 22, color: rgb(0.95, 0.97, 1) });
  page.drawText(text, { x: 55, y, size: 11, font: boldFont, color: rgb(0.04, 0.08, 0.28) });
  return y - 28;
}

/* ── Single image PDF ──────────────────────────────────────────────────────── */

async function downloadSinglePdf(
  result: any,
  modality: string | null,
  date: string,
  fileName: string | null, patient: { patientName: string; patientAge: string; patientGender: string }, setLoading: (v: boolean) => void,
) {
  setLoading(true);
  try {
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
    const boldFont = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

    const [overlayBytes, heatmapBytes, inputBytes] = await Promise.all([
      fetchImageAsBytes(result.image_urls.overlay),
      fetchImageAsBytes(result.image_urls.heatmap),
      fetchImageAsBytes(result.image_urls.input_gray),
    ]);

    const overlayImg = await embedImage(pdfDoc, overlayBytes);
    const heatmapImg = await embedImage(pdfDoc, heatmapBytes);
    const inputImg = await embedImage(pdfDoc, inputBytes);

    const isAneurysm = result.detection_prediction === 1;
    const prob = result.detection_probabilities.aneurysm;

    // ── Page 1: Header + Images ──────────────────────────────────────────
    const p1 = pdfDoc.addPage([612, 792]);

    p1.drawRectangle({ x: 0, y: 752, width: 612, height: 40, color: rgb(0.04, 0.08, 0.28) });
    p1.drawText("AERUX — Diagnostic Report", { x: 50, y: 762, size: 14, font: boldFont, color: rgb(1, 1, 1) });

    let y = 730;
    const statusColor = isAneurysm ? rgb(0.8, 0.1, 0.1) : rgb(0, 0.55, 0.3);
    p1.drawText(isAneurysm ? "ANEURYSM DETECTED" : "NO ANEURYSM DETECTED", {
      x: 50, y, size: 12, font: boldFont, color: statusColor,
    });
    y -= 20;

    const meta = [
      `Patient Name: ${patient.patientName || `P-${result.result_id?.toUpperCase().slice(0, 5) || "00000"}`}`,
      `Age/Sex: ${patient.patientAge || "U"}/${patient.patientGender ? patient.patientGender.charAt(0) : "U"}`,
      `Date: ${date}`,
      `Modality: ${modality || "Auto"}`,
      `File: ${fileName || "N/A"}`,
    ];
    p1.drawText(meta.join("   |   "), { x: 50, y, size: 8, font, color: rgb(0.4, 0.4, 0.4) });
    y -= 16;
    p1.drawText(`Detection Confidence: ${(prob * 100).toFixed(1)}%   |   Processing: ${result.processing_time_ms} ms`, {
      x: 50, y, size: 8, font, color: rgb(0.4, 0.4, 0.4),
    });
    y -= 24;

    y = drawSectionTitle(p1, font, boldFont, "Scan Imaging", y);

    const imgW = 240;
    const imgH = 200;

    if (inputImg) {
      p1.drawText("Original Input", { x: 55, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
      p1.drawImage(inputImg, { x: 55, y: y - imgH - 4, width: imgW, height: imgH });
    }
    if (overlayImg) {
      p1.drawText("AI Overlay", { x: 315, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
      p1.drawImage(overlayImg, { x: 315, y: y - imgH - 4, width: imgW, height: imgH });
    }
    y -= imgH + 20;

    if (heatmapImg) {
      p1.drawText("Probability Heatmap", { x: 55, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
      p1.drawImage(heatmapImg, { x: 55, y: y - imgH - 4, width: imgW, height: imgH });
    }
    y -= imgH + 24;

    // ── Page 2: Location assessment + summary ────────────────────────────
    const p2 = pdfDoc.addPage([612, 792]);
    p2.drawRectangle({ x: 0, y: 752, width: 612, height: 40, color: rgb(0.04, 0.08, 0.28) });
    p2.drawText("AERUX — Location Assessment", { x: 50, y: 762, size: 14, font: boldFont, color: rgb(1, 1, 1) });

    let y2 = 730;
    y2 = drawSectionTitle(p2, font, boldFont, "Anatomical Location Probabilities", y2);

    const locs = Object.entries(result.all_location_probabilities)
      .sort(([, a], [, b]) => (b as number) - (a as number)) as [string, number][];

    for (const [label, lp] of locs) {
      const pct = (lp * 100).toFixed(1);
      p2.drawText(label, { x: 60, y: y2, size: 9, font, color: rgb(0.2, 0.2, 0.2) });
      p2.drawText(`${pct}%`, { x: 350, y: y2, size: 9, font: boldFont, color: lp > 0.5 ? rgb(0.8, 0.1, 0.1) : rgb(0.04, 0.08, 0.28) });

      // Bar background
      p2.drawRectangle({ x: 400, y: y2 - 1, width: 150, height: 8, color: rgb(0.93, 0.93, 0.93) });
      // Bar fill
      const barColor = lp > 0.5 ? rgb(0.9, 0.2, 0.2) : rgb(0, 0.46, 0.85);
      p2.drawRectangle({ x: 400, y: y2 - 1, width: Math.max(1, 150 * lp), height: 8, color: barColor });
      y2 -= 18;
    }

    y2 -= 16;
    y2 = drawSectionTitle(p2, font, boldFont, "Clinical Summary", y2);
    const summaryLines = isAneurysm
      ? [
        `AI analysis detected a potential intracranial aneurysm with ${(prob * 100).toFixed(1)}% confidence.`,
        `Most probable location: ${locs[0]?.[0] || "Unknown"} (${((locs[0]?.[1] ?? 0) * 100).toFixed(1)}%).`,
      ]
      : [
        `AI analysis did not detect an intracranial aneurysm (confidence: ${(prob * 100).toFixed(1)}%).`,
        `Highest location probability: ${locs[0]?.[0] || "Unknown"} (${((locs[0]?.[1] ?? 0) * 100).toFixed(1)}%).`,
      ];
    for (const line of summaryLines) {
      p2.drawText(line, { x: 60, y: y2, size: 9, font, color: rgb(0.2, 0.2, 0.2) });
      y2 -= 16;
    }

    y2 -= 20;
    const disclaimer = "Disclaimer: This AI-generated report is for research/screening only. Not a medical diagnosis.";
    p2.drawText(disclaimer, { x: 50, y: y2, size: 7, font, color: rgb(0.55, 0.55, 0.55) });

    const bytes = await pdfDoc.save();
    triggerDownload(bytes, `AERUX-Report-${result.result_id || "scan"}.pdf`);
  } finally {
    setLoading(false);
  }
}

/* ── Series PDF ────────────────────────────────────────────────────────────── */

async function downloadSeriesPdf(
  result: SeriesResult,
  modality: string | null,
  date: string,
  fileName: string | null,
  patient: { patientName: string; patientAge: string; patientGender: string },
  setLoading: (v: boolean) => void,
) {
  setLoading(true);
  try {
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
    const boldFont = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

    const anyFlagged = result.flagged_windows > 0;
    const topW = result.top_results[0] as SeriesTopResult | undefined;

    // Fetch top window images
    let overlayImg = null;
    let heatmapImg = null;
    let inputImg = null;
    if (topW) {
      const [ob, hb, ib] = await Promise.all([
        fetchImageAsBytes(topW.image_urls.overlay),
        fetchImageAsBytes(topW.image_urls.heatmap),
        fetchImageAsBytes(topW.image_urls.input_gray),
      ]);
      overlayImg = await embedImage(pdfDoc, ob);
      heatmapImg = await embedImage(pdfDoc, hb);
      inputImg = await embedImage(pdfDoc, ib);
    }

    // ── Page 1 ───────────────────────────────────────────────────────────
    const p1 = pdfDoc.addPage([612, 792]);
    p1.drawRectangle({ x: 0, y: 752, width: 612, height: 40, color: rgb(0.04, 0.08, 0.28) });
    p1.drawText("AERUX — Series Scan Report", { x: 50, y: 762, size: 14, font: boldFont, color: rgb(1, 1, 1) });

    let y = 730;
    const sc = anyFlagged ? rgb(0.8, 0.1, 0.1) : rgb(0, 0.55, 0.3);
    p1.drawText(
      anyFlagged
        ? `POTENTIAL ANEURYSM — ${result.flagged_windows} flagged window(s)`
        : "NO ANEURYSM DETECTED",
      { x: 50, y, size: 12, font: boldFont, color: sc },
    );
    y -= 20;

    const meta = [
      `Patient Name: ${patient.patientName || `P-${result.result_id?.toUpperCase().slice(0, 5) || "00000"}`}`,
      `Age/Sex: ${patient.patientAge || "U"}/${patient.patientGender ? patient.patientGender.charAt(0) : "U"}`,
      `Date: ${date}`,
      `Modality: ${modality || "Auto"}`,
      `Slices: ${result.total_slices}`,
      `Windows: ${result.total_windows}`,
    ];
    p1.drawText(meta.join("  |  "), { x: 50, y, size: 8, font, color: rgb(0.4, 0.4, 0.4) });
    y -= 16;
    p1.drawText(
      `Peak probability: ${(result.max_aneurysm_prob * 100).toFixed(1)}%   |   File: ${fileName || "N/A"}   |   Processing: ${(result.processing_time_ms / 1000).toFixed(1)}s`,
      { x: 50, y, size: 8, font, color: rgb(0.4, 0.4, 0.4) },
    );
    y -= 24;

    if (topW) {
      y = drawSectionTitle(p1, font, boldFont, `Top Window — Slice ${topW.center_slice}`, y);

      const imgW = 240;
      const imgH = 200;

      if (inputImg) {
        p1.drawText("Original Input", { x: 55, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
        p1.drawImage(inputImg, { x: 55, y: y - imgH - 4, width: imgW, height: imgH });
      }
      if (overlayImg) {
        p1.drawText("AI Overlay", { x: 315, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
        p1.drawImage(overlayImg, { x: 315, y: y - imgH - 4, width: imgW, height: imgH });
      }
      y -= imgH + 20;

      if (heatmapImg) {
        p1.drawText("Probability Heatmap", { x: 55, y, size: 8, font: boldFont, color: rgb(0.2, 0.2, 0.2) });
        p1.drawImage(heatmapImg, { x: 55, y: y - imgH - 4, width: imgW, height: imgH });
      }
    }

    // ── Page 2: Location + summary ───────────────────────────────────────
    const p2 = pdfDoc.addPage([612, 792]);
    p2.drawRectangle({ x: 0, y: 752, width: 612, height: 40, color: rgb(0.04, 0.08, 0.28) });
    p2.drawText("AERUX — Location Assessment", { x: 50, y: 762, size: 14, font: boldFont, color: rgb(1, 1, 1) });

    let y2 = 730;

    if (topW) {
      y2 = drawSectionTitle(p2, font, boldFont, `Anatomical Locations (Slice ${topW.center_slice})`, y2);

      const locs = Object.entries(topW.all_location_probabilities).sort(
        ([, a], [, b]) => b - a,
      ) as [string, number][];

      for (const [label, lp] of locs) {
        p2.drawText(label, { x: 60, y: y2, size: 9, font, color: rgb(0.2, 0.2, 0.2) });
        p2.drawText(`${(lp * 100).toFixed(1)}%`, {
          x: 350, y: y2, size: 9, font: boldFont,
          color: lp > 0.5 ? rgb(0.8, 0.1, 0.1) : rgb(0.04, 0.08, 0.28),
        });
        p2.drawRectangle({ x: 400, y: y2 - 1, width: 150, height: 8, color: rgb(0.93, 0.93, 0.93) });
        p2.drawRectangle({
          x: 400, y: y2 - 1, width: Math.max(1, 150 * lp), height: 8,
          color: lp > 0.5 ? rgb(0.9, 0.2, 0.2) : rgb(0, 0.46, 0.85),
        });
        y2 -= 18;
      }
    }

    y2 -= 16;
    y2 = drawSectionTitle(p2, font, boldFont, "Clinical Summary", y2);
    const lines = anyFlagged
      ? [
        `Series scan: ${result.total_slices} slices, ${result.flagged_windows} flagged window(s).`,
        `Peak confidence: ${(result.max_aneurysm_prob * 100).toFixed(1)}% at slice ${topW?.center_slice ?? "N/A"}.`,
      ]
      : [
        `Series scan: ${result.total_slices} slices, no windows exceeded 50% threshold.`,
        `Highest probability: ${(result.max_aneurysm_prob * 100).toFixed(1)}%.`,
      ];
    for (const l of lines) {
      p2.drawText(l, { x: 60, y: y2, size: 9, font, color: rgb(0.2, 0.2, 0.2) });
      y2 -= 16;
    }

    y2 -= 20;
    p2.drawText(
      "Disclaimer: This AI-generated report is for research/screening only. Not a medical diagnosis.",
      { x: 50, y: y2, size: 7, font, color: rgb(0.55, 0.55, 0.55) },
    );

    const bytes = await pdfDoc.save();
    triggerDownload(bytes, `AERUX-Series-Report-${result.result_id || "scan"}.pdf`);
  } finally {
    setLoading(false);
  }
}

function triggerDownload(bytes: Uint8Array, filename: string) {
  const blob = new Blob([bytes as BlobPart], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}




