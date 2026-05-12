"use client";

import { useState } from "react";
import Link from "next/link";
import { Download, Printer, Upload } from "lucide-react";
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

  const patient = {
    name: patientName || `P-${(result?.result_id || seriesResult?.result_id || "0000").toUpperCase().slice(0, 5)}`,
    age: patientAge || "Unknown",
    gender: patientGender || "Unknown",
    mrn: `MRN-${Math.floor(100000 + Math.random() * 900000)}`,
  };

  if (seriesResult) {
    return <SeriesReport result={seriesResult} modality={modality} fileName={fileName} patient={patient} />;
  }
  if (result) {
    return <SingleReport result={result} modality={modality} fileName={fileName} patient={patient} />;
  }
  return <EmptyReport />;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Empty state
   ═══════════════════════════════════════════════════════════════════════════════ */

function EmptyReport() {
  return (
    <main className="relative mx-auto w-full max-w-4xl px-6 py-14">
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
   Helper: Format Date
   ═══════════════════════════════════════════════════════════════════════════════ */
function getTodayDate() {
  return new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
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
  patient: { name: string; age: string; gender: string; mrn: string };
}) {
  const isAneurysm = result.detection_prediction === 1;
  const prob = result.detection_probabilities.aneurysm;
  const todayDate = getTodayDate();

  const locations = Object.entries(result.all_location_probabilities)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 3) as [string, number][];

  const [downloading, setDownloading] = useState(false);

  const findingsText = isAneurysm
    ? `The AI algorithm evaluated the intracranial vascular structures for the presence of aneurysms. A high-confidence finding (${(prob * 100).toFixed(1)}%) consistent with an intracranial aneurysm was detected. The primary region of suspicion is the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] || 0) * 100).toFixed(1)}% probability). Secondary probabilistic mapping indicates involvement in the ${locations[1]?.[0] || "N/A"} (${((locations[1]?.[1] || 0) * 100).toFixed(1)}%) and the ${locations[2]?.[0] || "N/A"} (${((locations[2]?.[1] || 0) * 100).toFixed(1)}%).`
    : `The AI algorithm evaluated the intracranial vascular structures for the presence of aneurysms. No regions of high probability for intracranial aneurysm were detected. The maximum detection confidence across the analyzed plane was ${(prob * 100).toFixed(1)}%, which is below the clinical threshold for a positive finding. The highest probabilistic regions were the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] || 0) * 100).toFixed(1)}%), the ${locations[1]?.[0] || "N/A"} (${((locations[1]?.[1] || 0) * 100).toFixed(1)}%), and the ${locations[2]?.[0] || "N/A"} (${((locations[2]?.[1] || 0) * 100).toFixed(1)}%).`;

  const impression1 = isAneurysm
    ? `1. AI analysis indicates a HIGH probability (${(prob * 100).toFixed(1)}%) of intracranial aneurysm.`
    : `1. No evidence of intracranial aneurysm detected by automated analysis.`;

  return (
    <main className="relative mx-auto w-full max-w-4xl px-6 py-14 print:px-0 print:py-0">
      <ReportActions
        onPrint={() => window.print()}
        onDownload={() => downloadSinglePdf(result, modality, todayDate, fileName, patient, findingsText, impression1, setDownloading)}
        downloading={downloading}
      />
      <ReportDocument
        patient={patient}
        todayDate={todayDate}
        modality={modality}
        fileName={fileName}
        findingsText={findingsText}
        impression1={impression1}
        images={[
          { label: "Original Input", src: result.image_urls.input_gray },
          { label: "AI Overlay", src: result.image_urls.overlay },
        ]}
      />
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
  patient: { name: string; age: string; gender: string; mrn: string };
}) {
  const selectedSeriesSlice = useUIStore((s) => s.selectedSeriesSlice);
  const dynamicSlices = useUIStore((s) => s.dynamicSlices);

  const currentW = result.top_results.find((tr) => tr.center_slice === selectedSeriesSlice) ||
    (selectedSeriesSlice !== null ? dynamicSlices[selectedSeriesSlice] : null) ||
    result.top_results[0] as SeriesTopResult | undefined;

  const anyFlagged = result.flagged_windows > 0;
  const todayDate = getTodayDate();
  const [downloading, setDownloading] = useState(false);

  let locations: [string, number][] = [];
  if (currentW) {
    locations = Object.entries(currentW.all_location_probabilities)
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 3) as [string, number][];
  }

  const findingsText = anyFlagged && currentW
    ? `Automated volumetric analysis evaluated ${result.total_slices} slices (step size ${result.step}) covering the cerebral vasculature. The algorithm flagged ${result.flagged_windows} out of ${result.total_windows} analyzed windows. A high-confidence finding was detected at slice ${currentW.center_slice} with a peak probability of ${(result.max_aneurysm_prob * 100).toFixed(1)}%. The primary region of suspicion is the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] || 0) * 100).toFixed(1)}% probability). Secondary probabilistic mapping indicates involvement in the ${locations[1]?.[0] || "N/A"} (${((locations[1]?.[1] || 0) * 100).toFixed(1)}%) and the ${locations[2]?.[0] || "N/A"} (${((locations[2]?.[1] || 0) * 100).toFixed(1)}%).`
    : `Automated volumetric analysis evaluated ${result.total_slices} slices (step size ${result.step}) covering the cerebral vasculature. No regions of high probability for intracranial aneurysm were detected across all ${result.total_windows} windows. The peak detection confidence was ${(result.max_aneurysm_prob * 100).toFixed(1)}%, which remains below the clinical threshold for a positive finding. The highest probabilistic regions in the selected slice were the ${locations[0]?.[0] || "unknown region"} (${((locations[0]?.[1] || 0) * 100).toFixed(1)}%), the ${locations[1]?.[0] || "N/A"} (${((locations[1]?.[1] || 0) * 100).toFixed(1)}%), and the ${locations[2]?.[0] || "N/A"} (${((locations[2]?.[1] || 0) * 100).toFixed(1)}%).`;

  const impression1 = anyFlagged
    ? `1. AI analysis indicates a HIGH probability of intracranial aneurysm (Peak: ${(result.max_aneurysm_prob * 100).toFixed(1)}% at slice ${currentW?.center_slice}).`
    : `1. No evidence of intracranial aneurysm detected by automated volumetric analysis.`;

  const images = currentW
    ? [
        { label: `Original Input (Slice ${currentW.center_slice})`, src: currentW.image_urls.input_gray },
        { label: `AI Overlay (Slice ${currentW.center_slice})`, src: currentW.image_urls.overlay },
      ]
    : [];

  return (
    <main className="relative mx-auto w-full max-w-4xl px-6 py-14 print:px-0 print:py-0">
      <ReportActions
        onPrint={() => window.print()}
        onDownload={() => downloadSeriesPdf(result, currentW, modality, todayDate, fileName, patient, findingsText, impression1, setDownloading)}
        downloading={downloading}
      />
      <ReportDocument
        patient={patient}
        todayDate={todayDate}
        modality={modality}
        fileName={fileName}
        findingsText={findingsText}
        impression1={impression1}
        images={images}
      />
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Shared Report UI Components
   ═══════════════════════════════════════════════════════════════════════════════ */

function ReportActions({ onPrint, onDownload, downloading }: { onPrint: () => void; onDownload: () => void; downloading: boolean }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-6 print:hidden">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Report Preview</h1>
      </div>
      <div className="flex gap-2">
        <button
          onClick={onPrint}
          className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-sm font-medium text-[var(--color-aerux-navy)] shadow-sm ring-1 ring-black/10 hover:bg-zinc-50 transition"
        >
          <Printer className="h-4 w-4" /> Print
        </button>
        <button
          onClick={onDownload}
          disabled={downloading}
          className="inline-flex items-center gap-2 rounded-xl bg-[var(--color-aerux-navy)] px-4 py-2 text-sm font-medium text-white shadow transition hover:brightness-110 disabled:opacity-60"
        >
          <Download className="h-4 w-4" />
          {downloading ? "Generating PDF..." : "Download PDF"}
        </button>
      </div>
    </div>
  );
}

function ReportDocument({
  patient,
  todayDate,
  modality,
  fileName,
  findingsText,
  impression1,
  images,
}: {
  patient: { name: string; age: string; gender: string; mrn: string };
  todayDate: string;
  modality: string | null;
  fileName: string | null;
  findingsText: string;
  impression1: string;
  images: { label: string; src: string }[];
}) {
  return (
    <div className="bg-white p-12 shadow-sm ring-1 ring-black/5 rounded-xl print:shadow-none print:ring-0 print:p-0 font-serif text-zinc-900 max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b-2 border-zinc-900 pb-4 mb-6">
        <h1 className="text-2xl font-bold uppercase tracking-widest text-center">AERUX Neuroimaging</h1>
        <h2 className="text-lg font-semibold uppercase text-center mt-1">Diagnostic Radiology Report</h2>
      </div>

      {/* Demographics */}
      <div className="grid grid-cols-2 gap-4 text-sm mb-8">
        <div>
          <p><span className="font-semibold">Patient Name:</span> {patient.name}</p>
          <p><span className="font-semibold">MRN:</span> {patient.mrn}</p>
          <p><span className="font-semibold">Age/Sex:</span> {patient.age} / {patient.gender}</p>
        </div>
        <div>
          <p><span className="font-semibold">Exam Date:</span> {todayDate}</p>
          <p><span className="font-semibold">Report Date:</span> {todayDate}</p>
          <p><span className="font-semibold">Physician:</span> Referring Provider</p>
        </div>
      </div>

      {/* Exam Info */}
      <div className="space-y-6 text-sm">
        <section>
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-2">Exam</h3>
          <p>Automated Analysis of {modality || "Auto-detected"} Head/Brain (File: {fileName || "N/A"})</p>
        </section>

        <section>
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-2">Clinical Indication</h3>
          <p>Screening for suspected intracranial aneurysm.</p>
        </section>

        <section>
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-2">Technique</h3>
          <p>Automated AI-assisted volumetric and spatial analysis of cerebral vasculature via AERUX Deep Learning architecture.</p>
        </section>

        <section>
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-2">Findings</h3>
          <p className="leading-relaxed">{findingsText}</p>
        </section>

        <section>
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-2">Impression</h3>
          <p className="font-bold mb-1">{impression1}</p>
          <p className="font-bold">2. Findings are generated by an AI assistant and must be correlated with clinical history and formally verified by a board-certified neuroradiologist.</p>
        </section>
      </div>

      <div className="mt-12 text-center text-xs text-zinc-500 border-t border-zinc-200 pt-4">
        <p>Electronically generated by AERUX AI Assistant.</p>
        <p>Not a final medical diagnosis. For research and screening purposes only.</p>
      </div>

      {/* Key Images (Page Break in Print) */}
      {images.length > 0 && (
        <div className="mt-16 print:break-before-page">
          <h3 className="font-bold uppercase border-b border-zinc-200 pb-1 mb-4 text-sm">Key Images</h3>
          <div className="grid grid-cols-2 gap-6">
            {images.map((img, idx) => (
              <div key={idx} className="flex flex-col items-center">
                <img src={img.src} alt={img.label} className="w-full max-h-[300px] object-contain bg-black/5" />
                <p className="text-xs font-semibold mt-2">{img.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}
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

function wrapText(text: string, maxWidth: number, font: any, fontSize: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let currentLine = words[0];

  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const width = font.widthOfTextAtSize(currentLine + " " + word, fontSize);
    if (width < maxWidth) {
      currentLine += " " + word;
    } else {
      lines.push(currentLine);
      currentLine = word;
    }
  }
  lines.push(currentLine);
  return lines;
}

function drawParagraph(
  page: any,
  text: string,
  x: number,
  y: number,
  font: any,
  fontSize: number,
  maxWidth: number,
  lineHeight: number,
  color: any
) {
  const lines = wrapText(text, maxWidth, font, fontSize);
  let currentY = y;
  for (const line of lines) {
    page.drawText(line, { x, y: currentY, size: fontSize, font, color });
    currentY -= lineHeight;
  }
  return currentY;
}

function drawReportHeader(p: any, font: any, boldFont: any) {
  p.drawText("AERUX NEUROIMAGING", { x: 50, y: 740, size: 16, font: boldFont, color: rgb(0, 0, 0) });
  p.drawText("DIAGNOSTIC RADIOLOGY REPORT", { x: 50, y: 720, size: 12, font: boldFont, color: rgb(0, 0, 0) });
  p.drawLine({ start: { x: 50, y: 705 }, end: { x: 562, y: 705 }, thickness: 2, color: rgb(0, 0, 0) });
}

function drawDemographics(p: any, patient: any, date: string, font: any, boldFont: any, startY: number) {
  const col1X = 50;
  const col2X = 320;
  const ls = 14;
  let y = startY;

  const drawField = (label: string, value: string, x: number, yPos: number) => {
    p.drawText(`${label}:`, { x, y: yPos, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    p.drawText(value, { x: x + font.widthOfTextAtSize(`${label}: `, 10) + 5, y: yPos, size: 10, font: font, color: rgb(0, 0, 0) });
  };

  drawField("Patient Name", patient.name, col1X, y);
  drawField("Exam Date", date, col2X, y);
  y -= ls;
  drawField("MRN", patient.mrn, col1X, y);
  drawField("Report Date", date, col2X, y);
  y -= ls;
  drawField("Age/Sex", `${patient.age} / ${patient.gender}`, col1X, y);
  drawField("Physician", "Referring Provider", col2X, y);

  return y - 24;
}

function drawSection(p: any, title: string, text: string, y: number, font: any, boldFont: any, isImpression = false) {
  p.drawText(title.toUpperCase(), { x: 50, y, size: 10, font: boldFont, color: rgb(0, 0, 0) });
  p.drawLine({ start: { x: 50, y: y - 3 }, end: { x: 562, y: y - 3 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });
  y -= 16;
  
  const contentFont = isImpression ? boldFont : font;
  y = drawParagraph(p, text, 50, y, contentFont, 10, 512, 14, rgb(0, 0, 0));
  return y - 12;
}

/* ── Single image PDF ──────────────────────────────────────────────────────── */

async function downloadSinglePdf(
  result: any,
  modality: string | null,
  date: string,
  fileName: string | null, 
  patient: { name: string; age: string; gender: string; mrn: string }, 
  findingsText: string,
  impression1: string,
  setLoading: (v: boolean) => void,
) {
  setLoading(true);
  try {
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.TimesRoman);
    const boldFont = await pdfDoc.embedFont(StandardFonts.TimesRomanBold);

    const [overlayBytes, inputBytes] = await Promise.all([
      fetchImageAsBytes(result.image_urls.overlay),
      fetchImageAsBytes(result.image_urls.input_gray),
    ]);

    const overlayImg = await embedImage(pdfDoc, overlayBytes);
    const inputImg = await embedImage(pdfDoc, inputBytes);

    // ── Page 1: Main Report ──────────────────────────────────────────
    const p1 = pdfDoc.addPage([612, 792]);
    drawReportHeader(p1, font, boldFont);
    
    let y = 680;
    y = drawDemographics(p1, patient, date, font, boldFont, y);

    y = drawSection(p1, "Exam", `Automated Analysis of ${modality || "Auto-detected"} Head/Brain (File: ${fileName || "N/A"})`, y, font, boldFont);
    y = drawSection(p1, "Clinical Indication", "Screening for suspected intracranial aneurysm.", y, font, boldFont);
    y = drawSection(p1, "Technique", "Automated AI-assisted volumetric and spatial analysis of cerebral vasculature via AERUX Deep Learning architecture.", y, font, boldFont);
    y = drawSection(p1, "Findings", findingsText, y, font, boldFont);
    
    // Impression is multiline
    p1.drawText("IMPRESSION", { x: 50, y, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    p1.drawLine({ start: { x: 50, y: y - 3 }, end: { x: 562, y: y - 3 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });
    y -= 16;
    y = drawParagraph(p1, impression1, 50, y, boldFont, 10, 512, 14, rgb(0, 0, 0));
    y = drawParagraph(p1, "2. Findings are generated by an AI assistant and must be correlated with clinical history and formally verified by a board-certified neuroradiologist.", 50, y, boldFont, 10, 512, 14, rgb(0, 0, 0));

    // Disclaimer footer
    p1.drawLine({ start: { x: 50, y: 70 }, end: { x: 562, y: 70 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });
    p1.drawText("Electronically generated by AERUX AI Assistant. Not a final medical diagnosis. For research and screening purposes only.", { x: 50, y: 55, size: 8, font, color: rgb(0.4, 0.4, 0.4) });

    // ── Page 2: Key Images ──────────────────────────────────────────
    const p2 = pdfDoc.addPage([612, 792]);
    drawReportHeader(p2, font, boldFont);
    p2.drawText("KEY IMAGES", { x: 50, y: 680, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    p2.drawLine({ start: { x: 50, y: 677 }, end: { x: 562, y: 677 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });

    const imgW = 220;
    const imgH = 220;

    if (inputImg) {
      p2.drawImage(inputImg, { x: 60, y: 430, width: imgW, height: imgH });
      p2.drawText("Original Input", { x: 60, y: 415, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    }
    if (overlayImg) {
      p2.drawImage(overlayImg, { x: 332, y: 430, width: imgW, height: imgH });
      p2.drawText("AI Overlay", { x: 332, y: 415, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    }

    const bytes = await pdfDoc.save();
    triggerDownload(bytes, `AERUX-Report-${patient.mrn}.pdf`);
  } finally {
    setLoading(false);
  }
}

/* ── Series PDF ────────────────────────────────────────────────────────────── */

async function downloadSeriesPdf(
  result: SeriesResult,
  currentW: SeriesTopResult | undefined,
  modality: string | null,
  date: string,
  fileName: string | null,
  patient: { name: string; age: string; gender: string; mrn: string },
  findingsText: string,
  impression1: string,
  setLoading: (v: boolean) => void,
) {
  setLoading(true);
  try {
    const pdfDoc = await PDFDocument.create();
    const font = await pdfDoc.embedFont(StandardFonts.TimesRoman);
    const boldFont = await pdfDoc.embedFont(StandardFonts.TimesRomanBold);

    let overlayImg = null;
    let inputImg = null;
    if (currentW) {
      const [ob, ib] = await Promise.all([
        fetchImageAsBytes(currentW.image_urls.overlay),
        fetchImageAsBytes(currentW.image_urls.input_gray),
      ]);
      overlayImg = await embedImage(pdfDoc, ob);
      inputImg = await embedImage(pdfDoc, ib);
    }

    // ── Page 1: Main Report ──────────────────────────────────────────
    const p1 = pdfDoc.addPage([612, 792]);
    drawReportHeader(p1, font, boldFont);
    
    let y = 680;
    y = drawDemographics(p1, patient, date, font, boldFont, y);

    y = drawSection(p1, "Exam", `Automated Analysis of ${modality || "Auto-detected"} Head/Brain (File: ${fileName || "N/A"})`, y, font, boldFont);
    y = drawSection(p1, "Clinical Indication", "Screening for suspected intracranial aneurysm.", y, font, boldFont);
    y = drawSection(p1, "Technique", `Automated AI-assisted volumetric and spatial analysis of cerebral vasculature via AERUX Deep Learning architecture. Evaluated ${result.total_slices} slices.`, y, font, boldFont);
    y = drawSection(p1, "Findings", findingsText, y, font, boldFont);
    
    // Impression
    p1.drawText("IMPRESSION", { x: 50, y, size: 10, font: boldFont, color: rgb(0, 0, 0) });
    p1.drawLine({ start: { x: 50, y: y - 3 }, end: { x: 562, y: y - 3 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });
    y -= 16;
    y = drawParagraph(p1, impression1, 50, y, boldFont, 10, 512, 14, rgb(0, 0, 0));
    y = drawParagraph(p1, "2. Findings are generated by an AI assistant and must be correlated with clinical history and formally verified by a board-certified neuroradiologist.", 50, y, boldFont, 10, 512, 14, rgb(0, 0, 0));

    // Disclaimer footer
    p1.drawLine({ start: { x: 50, y: 70 }, end: { x: 562, y: 70 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });
    p1.drawText("Electronically generated by AERUX AI Assistant. Not a final medical diagnosis. For research and screening purposes only.", { x: 50, y: 55, size: 8, font, color: rgb(0.4, 0.4, 0.4) });

    // ── Page 2: Key Images ──────────────────────────────────────────
    if (currentW && (inputImg || overlayImg)) {
      const p2 = pdfDoc.addPage([612, 792]);
      drawReportHeader(p2, font, boldFont);
      p2.drawText(`KEY IMAGES (Slice ${currentW.center_slice})`, { x: 50, y: 680, size: 10, font: boldFont, color: rgb(0, 0, 0) });
      p2.drawLine({ start: { x: 50, y: 677 }, end: { x: 562, y: 677 }, thickness: 0.5, color: rgb(0.8, 0.8, 0.8) });

      const imgW = 220;
      const imgH = 220;

      if (inputImg) {
        p2.drawImage(inputImg, { x: 60, y: 430, width: imgW, height: imgH });
        p2.drawText("Original Input", { x: 60, y: 415, size: 10, font: boldFont, color: rgb(0, 0, 0) });
      }
      if (overlayImg) {
        p2.drawImage(overlayImg, { x: 332, y: 430, width: imgW, height: imgH });
        p2.drawText("AI Overlay", { x: 332, y: 415, size: 10, font: boldFont, color: rgb(0, 0, 0) });
      }
    }

    const bytes = await pdfDoc.save();
    triggerDownload(bytes, `AERUX-Series-Report-${patient.mrn}.pdf`);
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
