"use client";

import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import { CheckCircle2, Share, Download, AlertCircle } from "lucide-react";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";
import { useUIStore } from "@/store/ui";

export default function ReportsPage() {
  const result = useUIStore((s) => s.predictionResult);
  const modality = useUIStore((s) => s.selectedModality);

  // Findings from real predictions or empty state
  const findings = result
    ? result.top_3_locations.map((loc) => ({
        title: loc.label,
        confidence: Math.round(loc.score * 100),
        note: `Location probability: ${(loc.score * 100).toFixed(1)}%`,
      }))
    : [];

  const isAneurysm = result?.detection_prediction === 1;

  async function handleDownload() {
    const pdfDoc = await PDFDocument.create();
    const page = pdfDoc.addPage([612, 792]);
    const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

    const title = "Diagnostic Report – Aneurysm Detection & Analysis";
    page.drawText(title, { x: 50, y: 740, size: 16, font, color: rgb(0.04, 0.08, 0.28) });
    
    // Status text based on prediction
    const statusText = isAneurysm ? "Status: Aneurysm Detected" : "Status: Clear";
    const statusColor = isAneurysm ? rgb(0.8, 0.1, 0.1) : rgb(0, 0.6, 0.3);
    
    page.drawText(statusText, { x: 50, y: 720, size: 10, font, color: statusColor });

    const todayDate = new Date().toISOString().split('T')[0];
    page.drawText(`Patient ID: P-${result?.result_id || "00123"}`, { x: 50, y: 690, size: 10, font });
    page.drawText(`Scan Date: ${todayDate}`, { x: 220, y: 690, size: 10, font });
    page.drawText(`Modality: ${modality ?? "Unknown"}`, { x: 400, y: 690, size: 10, font });

    page.drawText(
      `Aneurysm Confidence: ${((result?.detection_probabilities?.aneurysm ?? 0) * 100).toFixed(1)}%`,
      { x: 50, y: 670, size: 10, font }
    );

    page.drawText("Key Findings (Top Locations):", { x: 50, y: 640, size: 12, font });
    let y = 620;
    
    if (findings.length === 0) {
      page.drawText("No analysis data available. Please upload a scan.", { x: 60, y, size: 10, font });
    } else {
      findings.slice(0, 3).forEach((f, i) => {
        page.drawText(`${i + 1}. ${f.title}  (Confidence: ${f.confidence}%)`, { x: 60, y, size: 10, font });
        y -= 16;
        page.drawText(`- ${f.note}`, { x: 70, y, size: 10, font, color: rgb(0.25, 0.25, 0.25) });
        y -= 20;
      });
    }

    const bytes = await pdfDoc.save();
    const blob = new Blob([bytes], { type: "application/pdf" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AERUX-Diagnostic-Report-${result?.result_id || "Demo"}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleShare() {
    const link = `${location.origin}/reports/public/${result?.result_id || "demo"}`;
    try {
      await navigator.clipboard.writeText(link);
      alert("Share link copied to clipboard.");
    } catch {
      alert(link);
    }
  }

  const container = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.12 } } };
  const item = { hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } } };

  return (
    <main className="relative mx-auto w-full max-w-6xl px-6 py-14">
      {/* Header card */}
      <motion.div
        initial={fadeUp.initial}
        whileInView={fadeUp.animate}
        viewport={{ once: true, amount: 0.2 }}
        transition={fadeUpTransition(0)}
        className="rounded-2xl bg-[color:var(--aerux-navy)] p-6 text-white shadow"
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h1 className="text-xl font-semibold inline-flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="inline-block">
              <path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
            Diagnostic Report – Aneurysm Detection & Analysis
          </h1>
          <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm font-medium ring-1 ${isAneurysm ? 'bg-red-500/15 text-red-400 ring-red-500/30' : 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/30'}`}>
            {isAneurysm ? <AlertCircle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />} 
            {isAneurysm ? 'Aneurysm Detected' : 'Clear'}
          </span>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-6 text-sm text-white/85">
          <div>Patient ID: <span className="font-medium">P-{result?.result_id?.toUpperCase() || "00123"}</span></div>
          <div>Scan Date: <span className="font-medium">{new Date().toISOString().split('T')[0]}</span></div>
          <div>Modality: <span className="font-medium">{modality || "Unknown"}</span></div>
          <div>Overall Confidence: <span className="font-medium">{result ? `${(result.detection_probabilities.aneurysm * 100).toFixed(1)}%` : 'N/A'}</span></div>
        </div>
        <div className="mt-5 flex flex-wrap gap-3">
          <button
            onClick={handleDownload}
            disabled={!result}
            className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 font-medium text-[color:var(--aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-100 transform transition hover:scale-[1.03] disabled:opacity-50 disabled:hover:scale-100 disabled:cursor-not-allowed"
          >
            <Download className="h-4 w-4" /> Download PDF
          </button>
          <button
            onClick={handleShare}
            disabled={!result}
            className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 font-medium text-[color:var(--aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-100 transform transition hover:scale-[1.03] disabled:opacity-50 disabled:hover:scale-100 disabled:cursor-not-allowed"
          >
            <Share className="h-4 w-4" /> Share Link
          </button>
        </div>
      </motion.div>

      {/* Key Findings */}
      <section className="mt-10">
        <h2 className="text-lg font-semibold tracking-tight text-[color:var(--aerux-navy)]/90">Key Top Locations</h2>
        
        {!result ? (
          <div className="mt-4 p-8 text-center bg-zinc-50 rounded-2xl border border-zinc-200">
             <p className="text-zinc-500">No report data generated yet. Please upload and analyze a scan first.</p>
          </div>
        ) : (
          <motion.ul
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, amount: 0.2 }}
            className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
          >
            {findings.map((f, idx) => (
              <motion.li
                key={idx}
                variants={item}
                className="rounded-2xl bg-white p-5 ring-1 ring-black/10 transition hover:-translate-y-1 hover:shadow"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-[color:var(--aerux-navy)] font-semibold">{f.title}</h3>
                    <p className="mt-2 text-sm text-zinc-700">{f.note}</p>
                  </div>
                  <span className={`rounded-full px-3 py-1 text-sm font-medium ring-1 whitespace-nowrap ${f.confidence > 50 ? 'bg-red-50 text-red-700 ring-red-500/30' : 'bg-blue-50 text-[color:var(--aerux-navy)] ring-[color:var(--aerux-blue)]/30'}`}>
                    {f.confidence}%
                  </span>
                </div>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </section>
    </main>
  );
}
