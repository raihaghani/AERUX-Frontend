import { create } from "zustand";

type PredictionResult = {
  result_id: string;
  detection_prediction: 0 | 1;
  detection_probabilities: { no_aneurysm: number; aneurysm: number };
  top_3_locations: Array<{ label: string; score: number }>;
  all_location_probabilities: Record<string, number>;
  image_urls: {
    overlay: string;
    heatmap: string;
    input_gray: string;
    highlight_mask: string;
  };
  processing_time_ms: number;
};

export type SeriesSliceResult = {
  center_slice: number;
  aneurysm_prob: number;
};

export type SeriesTopResult = {
  center_slice: number;
  detection_prediction: 0 | 1;
  detection_probabilities: { no_aneurysm: number; aneurysm: number };
  top_3_locations: Array<{ label: string; score: number }>;
  all_location_probabilities: Record<string, number>;
  image_urls: {
    overlay: string;
    heatmap: string;
    input_gray: string;
    highlight_mask: string;
  };
};

export type SeriesResult = {
  result_id: string;
  total_slices: number;
  total_windows: number;
  step: number;
  max_aneurysm_prob: number;
  flagged_windows: number;
  slice_results: SeriesSliceResult[];
  top_results: SeriesTopResult[];
  processing_time_ms: number;
};

type UIState = {
  isUploadModalOpen: boolean;
  uploadProgress: number;
  selectedFileName: string | null;
  isLoading: boolean;
  selectedModality: "CTA" | "MRA" | "MRI" | null;
  predictionResult: PredictionResult | null;
  seriesResult: SeriesResult | null;
  predictionError: string | null;

  patientName: string;
  patientAge: string;
  patientGender: string;

  selectedSeriesSlice: number | null;
  dynamicSlices: Record<number, SeriesTopResult>;

  setUploadProgress: (value: number) => void;
  setSelectedFileName: (name: string | null) => void;
  setLoading: (value: boolean) => void;
  openUploadModal: () => void;
  closeUploadModal: () => void;

  setSelectedModality: (m: "CTA" | "MRA" | "MRI") => void;
  setPredictionResult: (r: PredictionResult | null) => void;
  setSeriesResult: (r: SeriesResult | null) => void;
  setPredictionError: (e: string | null) => void;

  setPatientName: (name: string) => void;
  setPatientAge: (age: string) => void;
  setPatientGender: (gender: string) => void;

  setSelectedSeriesSlice: (slice: number | null) => void;
  setDynamicSlices: (slices: Record<number, SeriesTopResult> | ((prev: Record<number, SeriesTopResult>) => Record<number, SeriesTopResult>)) => void;
};

export const useUIStore = create<UIState>((set) => ({
  isUploadModalOpen: false,
  uploadProgress: 0,
  selectedFileName: null,
  isLoading: false,
  selectedModality: null,
  predictionResult: null,
  seriesResult: null,
  predictionError: null,

  patientName: "",
  patientAge: "",
  patientGender: "",

  selectedSeriesSlice: null,
  dynamicSlices: {},

  setUploadProgress: (value) => set({ uploadProgress: value }),
  setSelectedFileName: (name) => set({ selectedFileName: name }),
  setLoading: (value) => set({ isLoading: value }),
  openUploadModal: () => set({ isUploadModalOpen: true }),
  closeUploadModal: () => set({ isUploadModalOpen: false }),

  setSelectedModality: (m) => set({ selectedModality: m }),
  setPredictionResult: (r) => set({ predictionResult: r }),
  setSeriesResult: (r) => {
    set({ seriesResult: r });
    if (r && r.top_results.length > 0) {
      set({ selectedSeriesSlice: r.top_results[0].center_slice });
    } else {
      set({ selectedSeriesSlice: null });
    }
  },
  setPredictionError: (e) => set({ predictionError: e }),

  setPatientName: (name) => set({ patientName: name }),
  setPatientAge: (age) => set({ patientAge: age }),
  setPatientGender: (gender) => set({ patientGender: gender }),

  setSelectedSeriesSlice: (slice) => set({ selectedSeriesSlice: slice }),
  setDynamicSlices: (updater) =>
    set((state) => ({
      dynamicSlices: typeof updater === "function" ? updater(state.dynamicSlices) : updater,
    })),
}));


