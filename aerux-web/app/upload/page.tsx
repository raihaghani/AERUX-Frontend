"use client";

import { useRef, useState, useEffect, useCallback, DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { fadeUp } from "@/components/motion/presets";
import {
  Upload,
  FileImage,
  Layers,
  CheckCircle2,
  Loader2,
  Circle,
  FileType2,
  Clock,
} from "lucide-react";
import { useUIStore } from "@/store/ui";
import { useRouter } from "next/navigation";

type ScanMode = "single" | "series";
type ProcessingStage =
  | null
  | "uploading"
  | "detecting"
  | "analyzing"
  | "results";

const PIPELINE_STEPS = [
  { key: "uploading", label: "Uploading file" },
  { key: "detecting", label: "Detecting modality" },
  { key: "analyzing", label: "Analyzing scan" },
  { key: "results", label: "Generating results" },
] as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [scanMode, setScanMode] = useState<ScanMode>("single");
  const [step, setStep] = useState(3);
  const [fastMode, setFastMode] = useState(false);

  const uploadProgress = useUIStore((s) => s.uploadProgress);
  const setUploadProgress = useUIStore((s) => s.setUploadProgress);
  const selectedFileName = useUIStore((s) => s.selectedFileName);
  const setSelectedFileName = useUIStore((s) => s.setSelectedFileName);
  const setSelectedModality = useUIStore((s) => s.setSelectedModality);
  const setPredictionResult = useUIStore((s) => s.setPredictionResult);
  const setSeriesResult = useUIStore((s) => s.setSeriesResult);
  const predictionError = useUIStore((s) => s.predictionError);
  const setPredictionError = useUIStore((s) => s.setPredictionError);

  const patientName = useUIStore((s) => s.patientName);
  const setPatientName = useUIStore((s) => s.setPatientName);
  const patientAge = useUIStore((s) => s.patientAge);
  const setPatientAge = useUIStore((s) => s.setPatientAge);
  const patientGender = useUIStore((s) => s.patientGender);
  const setPatientGender = useUIStore((s) => s.setPatientGender);

  const [stage, setStage] = useState<ProcessingStage>(null);
  const [fileSize, setFileSize] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!stage) {
      setElapsed(0);
      return;
    }
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, [stage]);

  const resetProcessing = useCallback(() => {
    setStage(null);
    setElapsed(0);
    setUploadProgress(0);
  }, [setUploadProgress]);

  const onBrowse = () => inputRef.current?.click();

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setSelectedFileName(file.name);
    setFileSize(file.size);

    if (scanMode === "series") {
      uploadSeries(file);
    } else {
      uploadFile(file);
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  function uploadFile(file: File) {
    const formData = new FormData();
    formData.append("file", file);

    setPredictionError(null);
    setPredictionResult(null);
    setSeriesResult(null);
    setStage("uploading");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.upload.onload = () => {
      setUploadProgress(100);
      setStage("detecting");
      setTimeout(() => setStage("analyzing"), 1200);
    };

    xhr.onload = () => {
      setStage("results");
      try {
        const res = JSON.parse(xhr.responseText || "{}");
        if (res.ok) {
          if (res.detected_modality) setSelectedModality(res.detected_modality);
          setPredictionResult(res);
          setTimeout(() => {
            resetProcessing();
            router.push("/visuals");
          }, 600);
        } else {
          setPredictionError(res.error ?? "Inference failed");
          resetProcessing();
        }
      } catch {
        setPredictionError("Failed to parse response");
        resetProcessing();
      }
    };

    xhr.onerror = () => {
      setPredictionError("Network error during upload");
      resetProcessing();
    };

    xhr.send(formData);
  }

  function uploadSeries(file: File) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("step", String(step));
    if (fastMode) formData.append("fast_mode", "true");

    setPredictionError(null);
    setPredictionResult(null);
    setSeriesResult(null);
    setStage("uploading");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload-series");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.upload.onload = () => {
      setUploadProgress(100);
      setStage("detecting");
      setTimeout(() => setStage("analyzing"), 1500);
    };

    xhr.onload = () => {
      setStage("results");
      try {
        const res = JSON.parse(xhr.responseText || "{}");
        if (res.ok) {
          if (res.detected_modality) setSelectedModality(res.detected_modality);
          setSeriesResult(res);
          setTimeout(() => {
            resetProcessing();
            router.push("/visuals");
          }, 600);
        } else {
          setPredictionError(res.error ?? "Series scan failed");
          resetProcessing();
        }
      } catch {
        setPredictionError("Failed to parse response");
        resetProcessing();
      }
    };

    xhr.onerror = () => {
      setPredictionError("Network error during upload");
      resetProcessing();
    };

    xhr.send(formData);
  }

  const acceptFilter =
    scanMode === "series"
      ? ".zip"
      : ".npy,.zip,.dcm,.nii,.nii.gz,.png,.jpg,.jpeg";

  const isProcessing = stage !== null;

  return (
    <main className="relative mx-auto w-full max-w-4xl px-6 py-14 min-h-screen aerux-dot-grid bg-[var(--color-surface-1)]">
      <h1 className="text-3xl font-bold tracking-tight">Upload Scan</h1>
      <p className="mt-2 text-zinc-700">
        Analyze a single image or scan a full DICOM series for aneurysm
        detection.
      </p>

      {/* Mode toggle */}
      {!isProcessing && (
        <div className="mt-6 inline-flex rounded-xl bg-[var(--color-surface-2)] p-1 ring-1 ring-black/5">
          <button
            type="button"
            onClick={() => setScanMode("single")}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${scanMode === "single"
              ? "bg-white text-[var(--color-aerux-navy)] shadow-sm"
              : "text-zinc-500 hover:text-zinc-700"
              }`}
          >
            <FileImage className="h-4 w-4" /> Single Image
          </button>
          <button
            type="button"
            onClick={() => setScanMode("series")}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${scanMode === "series"
              ? "bg-white text-[var(--color-aerux-navy)] shadow-sm"
              : "text-zinc-500 hover:text-zinc-700"
              }`}
          >
            <Layers className="h-4 w-4" /> Series Scan
          </button>
        </div>
      )}

      <AnimatePresence mode="wait">
        {isProcessing ? (
          <motion.div
            key="processing"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35 }}
            className="mt-6 rounded-2xl bg-white p-8 ring-1 ring-black/10 shadow-sm"
          >
            <ProcessingCard
              fileName={selectedFileName}
              fileSize={fileSize}
              stage={stage}
              uploadProgress={uploadProgress}
              elapsed={elapsed}
              isSeries={scanMode === "series"}
            />
          </motion.div>
        ) : (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35 }}
            className="mt-6 space-y-6"
          >
            {/* Patient Context Form */}
            <div className="rounded-2xl bg-white p-6 ring-1 ring-black/10 shadow-sm flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <label className="mb-1.5 block text-sm font-semibold text-[color:var(--aerux-navy)]">
                  Patient Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. John Doe / ID-4829"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-1)] px-4 py-2.5 text-sm outline-none focus:border-[var(--color-aerux-accent)] focus:bg-white focus:ring-2 focus:ring-[var(--color-aerux-accent-lt)]"
                />
              </div>
              <div className="w-full sm:w-28">
                <label className="mb-1.5 block text-sm font-semibold text-[color:var(--aerux-navy)]">
                  Age
                </label>
                <input
                  type="text"
                  placeholder="e.g. 45"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-1)] px-4 py-2.5 text-sm outline-none focus:border-[var(--color-aerux-accent)] focus:bg-white focus:ring-2 focus:ring-[var(--color-aerux-accent-lt)]"
                />
              </div>
              <div className="w-full sm:w-40">
                <label className="mb-1.5 block text-sm font-semibold text-[color:var(--aerux-navy)]">
                  Gender
                </label>
                <select
                  value={patientGender}
                  onChange={(e) => setPatientGender(e.target.value)}
                  className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-1)] px-4 py-[11px] text-sm outline-none focus:border-[var(--color-aerux-accent)] focus:bg-white focus:ring-2 focus:ring-[var(--color-aerux-accent-lt)]"
                >
                  <option value="">Select...</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            {/* Dropzone */}
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className="rounded-2xl bg-white p-8 ring-1 ring-black/10 shadow-sm transition-shadow"
              style={{
                boxShadow: isDragging
                  ? "0 0 0 3px rgba(0,116,217,0.3)"
                  : undefined,
              }}
            >
              <div className="flex flex-col items-center justify-center text-center">
                <span className="grid h-16 w-16 place-items-center rounded-2xl bg-white ring-1 ring-[var(--color-aerux-navy)] text-[var(--color-aerux-navy)] shadow-sm">
                  <Upload className="h-7 w-7" />
                </span>

                {scanMode === "single" ? (
                  <>
                    <p className="mt-4 text-lg font-medium text-[var(--color-aerux-navy)]">
                      Drop or select a medical image
                    </p>
                    <p className="mt-1 text-sm text-zinc-600 mb-3">
                      Preprocessed (.npy) · DICOM series (.zip) · Single image
                      (.dcm, .nii, .png, .jpg)
                    </p>
                    <p className="mb-4 max-w-lg rounded-lg bg-amber-50 px-4 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
                      <strong>Tip:</strong> Preprocessed .npy or a full DICOM
                      series (.zip) give the most accurate results. Single 2D
                      images (.dcm, .png) lack the multi-slice depth context the
                      model was trained on.
                    </p>
                  </>
                ) : (
                  <>
                    <p className="mt-4 text-lg font-medium text-[var(--color-aerux-navy)]">
                      Drop or select a DICOM series archive
                    </p>
                    <p className="mt-1 text-sm text-zinc-600 mb-6">
                      Upload a .zip containing DICOM (.dcm) files for a full
                      sliding-window scan
                    </p>
                  </>
                )}

                <div className="flex flex-col items-center gap-4">
                  {scanMode === "series" && (
                    <div className="flex flex-wrap items-center justify-center gap-4">
                      <label className="flex items-center gap-2 text-sm text-zinc-700">
                        <span className="font-medium">Step:</span>
                        <select
                          value={step}
                          onChange={(e) => setStep(Number(e.target.value))}
                          title="Select window step size"
                          className="rounded-lg border px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-[var(--color-aerux-accent)] bg-white cursor-pointer"
                        >
                          <option value={1}>1 (every slice)</option>
                          <option value={2}>2</option>
                          <option value={3}>3 (balanced)</option>
                          <option value={5}>5 (fast)</option>
                        </select>
                      </label>

                      <label className="flex items-center gap-2 text-sm text-zinc-700 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={fastMode}
                          onChange={(e) => setFastMode(e.target.checked)}
                          className="h-4 w-4 rounded border-zinc-300 text-[var(--color-aerux-accent)] focus:ring-[var(--color-aerux-accent)]"
                        />
                        <span>
                          Fast mode{" "}
                          <span className="text-xs text-zinc-500">
                            (skip N4 correction)
                          </span>
                        </span>
                      </label>
                    </div>
                  )}

                  <div>
                    <button
                      type="button"
                      onClick={onBrowse}
                      className="inline-flex h-11 items-center justify-center rounded-2xl bg-[var(--color-aerux-accent)] px-6 font-medium text-white shadow transition hover:brightness-105 transform hover:scale-[1.03]"
                    >
                      Select File
                    </button>
                    <input
                      ref={inputRef}
                      type="file"
                      accept={acceptFilter}
                      title="Choose a file to upload"
                      onChange={(e) => handleFiles(e.target.files)}
                      className="hidden"
                    />
                  </div>
                </div>

                {predictionError && (
                  <div className="mt-4 text-sm text-red-500 font-medium">
                    Error: {predictionError}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}

/* ─── Processing card ─────────────────────────────────────────────────────── */

function ProcessingCard({
  fileName,
  fileSize,
  stage,
  uploadProgress,
  elapsed,
  isSeries,
}: {
  fileName: string | null;
  fileSize: number;
  stage: ProcessingStage;
  uploadProgress: number;
  elapsed: number;
  isSeries: boolean;
}) {
  const stageIdx = PIPELINE_STEPS.findIndex((s) => s.key === stage);

  return (
    <div className="flex flex-col items-center gap-8">
      {/* File info */}
      <div className="flex items-center gap-3 rounded-xl bg-zinc-50 px-5 py-3 ring-1 ring-black/5">
        <FileType2 className="h-8 w-8 text-[var(--color-aerux-navy)] shrink-0" />
        <div className="text-left min-w-0">
          <p className="text-sm font-semibold text-[var(--color-aerux-navy)] truncate max-w-[280px]">
            {fileName ?? "file"}
          </p>
          <p className="text-xs text-zinc-500">
            {formatBytes(fileSize)} · {isSeries ? "Series Scan" : "Single Image"}
          </p>
        </div>
      </div>

      {/* Upload progress (only during upload stage) */}
      {stage === "uploading" && (
        <div className="w-full max-w-sm">
          <div className="flex items-center justify-between text-xs text-zinc-500 mb-1.5">
            <span>Uploading...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-zinc-200 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${uploadProgress}%` }}
              transition={{ ease: "easeOut" as const, duration: 0.3 }}
              className="h-full rounded-full bg-[var(--color-aerux-accent)]"
            />
          </div>
        </div>
      )}

      {/* Pipeline stepper */}
      <div className="w-full max-w-md">
        <div className="flex flex-col gap-0">
          {PIPELINE_STEPS.map((s, i) => {
            let status: "done" | "active" | "pending" = "pending";
            if (i < stageIdx) status = "done";
            else if (i === stageIdx) status = "active";

            return (
              <div key={s.key} className="flex items-center gap-3 py-2">
                <StepIcon status={status} />
                <span
                  className={`text-sm font-medium transition-colors duration-300 ${status === "done"
                    ? "text-emerald-600"
                    : status === "active"
                      ? "text-[var(--color-aerux-navy)]"
                      : "text-zinc-400"
                    }`}
                >
                  {s.label}
                </span>
                {status === "active" && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="ml-auto text-xs text-zinc-400"
                  >
                    {formatTime(elapsed)}
                  </motion.span>
                )}
                {status === "done" && (
                  <span className="ml-auto text-xs text-emerald-500">Done</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Elapsed timer */}
      <div className="flex items-center gap-1.5 text-xs text-zinc-400">
        <Clock className="h-3.5 w-3.5" />
        <span>Elapsed: {formatTime(elapsed)}</span>
      </div>
    </div>
  );
}

/* ─── Step icon ───────────────────────────────────────────────────────────── */

function StepIcon({ status }: { status: "done" | "active" | "pending" }) {
  if (status === "done") {
    return (
      <motion.span
        initial={{ scale: 0.5 }}
        animate={{ scale: 1 }}
        className="shrink-0"
      >
        <CheckCircle2 className="h-5 w-5 text-emerald-500" />
      </motion.span>
    );
  }
  if (status === "active") {
    return <Loader2 className="h-5 w-5 shrink-0 animate-spin text-[var(--color-aerux-accent)]" />;
  }
  return <Circle className="h-5 w-5 shrink-0 text-zinc-300" />;
}



