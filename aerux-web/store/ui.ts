import { create } from "zustand";

type PredictionResult = {
  result_id:                  string;
  detection_prediction:       0 | 1;
  detection_probabilities:    { no_aneurysm: number; aneurysm: number };
  top_3_locations:            Array<{ label: string; score: number }>;
  all_location_probabilities: Record<string, number>;
  image_urls: {
    overlay:        string;
    heatmap:        string;
    input_gray:     string;
    highlight_mask: string;
  };
  processing_time_ms: number;
};

export type SeriesSliceResult = {
  center_slice: number;
  aneurysm_prob: number;
};

export type SeriesTopResult = {
  center_slice:               number;
  detection_prediction:       0 | 1;
  detection_probabilities:    { no_aneurysm: number; aneurysm: number };
  top_3_locations:            Array<{ label: string; score: number }>;
  all_location_probabilities: Record<string, number>;
  image_urls: {
    overlay:        string;
    heatmap:        string;
    input_gray:     string;
    highlight_mask: string;
  };
};

export type SeriesResult = {
  result_id:          string;
  total_slices:       number;
  total_windows:      number;
  step:               number;
  max_aneurysm_prob:  number;
  flagged_windows:    number;
  slice_results:      SeriesSliceResult[];
  top_results:        SeriesTopResult[];
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

  setUploadProgress: (value: number) => void;
  setSelectedFileName: (name: string | null) => void;
  setLoading: (value: boolean) => void;
  openUploadModal: () => void;
  closeUploadModal: () => void;

  setSelectedModality: (m: "CTA" | "MRA" | "MRI") => void;
  setPredictionResult: (r: PredictionResult | null) => void;
  setSeriesResult: (r: SeriesResult | null) => void;
  setPredictionError: (e: string | null) => void;
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

  setUploadProgress: (value) => set({ uploadProgress: value }),
  setSelectedFileName: (name) => set({ selectedFileName: name }),
  setLoading: (value) => set({ isLoading: value }),
  openUploadModal: () => set({ isUploadModalOpen: true }),
  closeUploadModal: () => set({ isUploadModalOpen: false }),

  setSelectedModality: (m) => set({ selectedModality: m }),
  setPredictionResult: (r) => set({ predictionResult: r }),
  setSeriesResult: (r) => set({ seriesResult: r }),
  setPredictionError: (e) => set({ predictionError: e }),
}));


