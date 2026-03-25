"use client";

import { useRef, useState, DragEvent } from "react";
import { motion } from "framer-motion";
import { fadeUp, fadeUpTransition } from "@/components/motion/presets";
import { Upload } from "lucide-react";
import { useUIStore } from "@/store/ui";
import { useRouter } from "next/navigation";

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const uploadProgress = useUIStore((s) => s.uploadProgress);
  const setUploadProgress = useUIStore((s) => s.setUploadProgress);
  const selectedFileName = useUIStore((s) => s.selectedFileName);
  const setSelectedFileName = useUIStore((s) => s.setSelectedFileName);
  const setSelectedModality = useUIStore((s) => s.setSelectedModality);
  const setPredictionResult = useUIStore((s) => s.setPredictionResult);
  const setPredictionError = useUIStore((s) => s.setPredictionError);
  const setLoading = useUIStore((s) => s.setLoading);
  
  const [modality, setModality] = useState<"CTA" | "MRA" | "MRI">("CTA");

  const onBrowse = () => inputRef.current?.click();

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setSelectedFileName(file.name);
    uploadFile(file);
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
    formData.append("modality", modality);

    setSelectedModality(modality);
    setPredictionError(null);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onloadstart = () => setLoading(true);

    xhr.onload = () => {
      setUploadProgress(100);
      try {
        const res = JSON.parse(xhr.responseText || "{}");
        if (res.ok) {
          setPredictionResult(res);
          setLoading(false);
          router.push("/visuals");
        } else {
          setPredictionError(res.error ?? "Inference failed");
          setLoading(false);
        }
      } catch {
        setPredictionError("Failed to parse response");
        setLoading(false);
      }
      setTimeout(() => setUploadProgress(0), 800);
    };

    xhr.onerror = () => {
      setPredictionError("Network error during upload");
      setLoading(false);
    };

    xhr.send(formData);
  }

  return (
    <main className="relative mx-auto w-full max-w-4xl px-6 py-14">
      <h1 className="text-3xl font-bold tracking-tight">Upload Scan</h1>
      <p className="mt-2 text-zinc-700">
        Formats supported: DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG
      </p>

      <motion.div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        initial={fadeUp.initial}
        whileInView={fadeUp.animate}
        viewport={{ once: true, amount: 0.2 }}
        transition={fadeUpTransition(0)}
        animate={{ scale: isDragging ? 1.02 : 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
        className="mt-8 rounded-2xl bg-white p-8 ring-1 ring-black/10 shadow-sm"
        style={{ boxShadow: isDragging ? "0 0 0 3px rgba(0,116,217,0.3)" : undefined }}
      >
        <div className="flex flex-col items-center justify-center text-center">
          <span className="grid h-16 w-16 place-items-center rounded-2xl bg-white ring-1 ring-[color:var(--aerux-navy)] text-[color:var(--aerux-navy)] shadow-sm">
            <Upload className="h-7 w-7" />
          </span>
          <p className="mt-4 text-lg font-medium text-[color:var(--aerux-navy)]">Drop or select medical image</p>
          <p className="mt-1 text-sm text-zinc-600 mb-6">
            DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG
          </p>

          <div className="flex flex-col items-center gap-4">
            <select
              value={modality}
              onChange={(e) => setModality(e.target.value as "CTA" | "MRA" | "MRI")}
              className="rounded-xl border px-3 py-2 text-sm font-medium text-[color:var(--aerux-navy)] outline-none focus:ring-2 focus:ring-[color:var(--aerux-accent)] bg-white cursor-pointer"
            >
              <option value="CTA">CTA — CT Angiography</option>
              <option value="MRA">MRA — MR Angiography</option>
              <option value="MRI">MRI — Magnetic Resonance</option>
            </select>

            <div>
              <button
                type="button"
                onClick={onBrowse}
                className="inline-flex h-11 items-center justify-center rounded-2xl bg-[color:var(--aerux-accent)] px-6 font-medium text-white shadow transition hover:brightness-105 transform hover:scale-[1.03]"
              >
                Select File
              </button>
              <input
                ref={inputRef}
                type="file"
                accept=".dcm,.nii,.nii.gz,.png"
                onChange={(e) => handleFiles(e.target.files)}
                className="hidden"
              />
            </div>
          </div>

          {selectedFileName && (
            <div className="mt-8 w-full max-w-xl">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-[color:var(--aerux-navy)]">{selectedFileName}</span>
                <span className="text-zinc-600">{uploadProgress}%</span>
              </div>
              <div className="mt-2 h-2 w-full rounded-full bg-zinc-200">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${uploadProgress}%` }}
                  className="h-2 rounded-full bg-[color:var(--aerux-accent)]"
                />
              </div>
            </div>
          )}
          
          {useUIStore((s) => s.predictionError) && (
            <div className="mt-4 text-sm text-red-500 font-medium">
              Error: {useUIStore((s) => s.predictionError)}
            </div>
          )}
        </div>
      </motion.div>
    </main>
  );
}
