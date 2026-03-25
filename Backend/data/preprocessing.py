"""
Medical Image Preprocessing Module for Aneurysm Detection

This module contains preprocessing techniques for different imaging modalities:
- MRI/MRA: N4 Bias Field Correction, Sato Vesselness Filter, Otsu Thresholding
- CTA: CLAHE, HU Windowing, Sato Vesselness Filter

Author: Refactored from original notebooks
Date: 2025
"""

import numpy as np
import cv2
import SimpleITK as sitk
from skimage import filters
from skimage.filters import frangi, sato
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MRIPreprocessor:
    """
    Preprocessing pipeline for MRI/MRA images.
    
    Pipeline:
        1. N4 Bias Field Correction - Corrects intensity non-uniformity
        2. Sato Vesselness Filter - Enhances tubular structures (blood vessels)
        3. Otsu Thresholding (optional) - Automatic segmentation
    """
    
    def __init__(
        self,
        apply_n4: bool = True,
        apply_vesselness: bool = True,
        apply_otsu: bool = False,
        vesselness_scale_range: Tuple[int, int] = (1, 8),
        vesselness_scale_step: int = 2,
        n4_max_iterations: list = None
    ):
        """
        Initialize MRI/MRA preprocessor.
        
        Args:
            apply_n4: Whether to apply N4 bias field correction
            apply_vesselness: Whether to apply Sato vesselness filter
            apply_otsu: Whether to apply Otsu thresholding
            vesselness_scale_range: Scale range for vesselness filter (min, max)
            vesselness_scale_step: Step size for scale range
            n4_max_iterations: Maximum iterations for N4 correction at each level
        """
        self.apply_n4 = apply_n4
        self.apply_vesselness = apply_vesselness
        self.apply_otsu = apply_otsu
        self.vesselness_scale_range = vesselness_scale_range
        self.vesselness_scale_step = vesselness_scale_step
        self.n4_max_iterations = n4_max_iterations or [50, 50, 50, 50]
    
    def n4_bias_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply N4 Bias Field Correction to remove intensity non-uniformity.
        
        N4 is an improved version of the nonparametric nonuniform intensity
        normalization (N3) method. It corrects low-frequency intensity
        variations in MRI images caused by magnetic field inhomogeneities.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Bias-corrected image with same shape as input
        """
        try:
            # Normalize to [0, 1] range
            img_normalized = self._normalize_image(image)
            
            # Convert to SimpleITK image (required for N4 filter)
            sitk_img = sitk.GetImageFromArray(img_normalized.astype(np.float32))
            
            # Create N4 bias field corrector
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations(self.n4_max_iterations)
            corrector.SetConvergenceThreshold(0.001)
            
            # Execute correction
            corrected_img = corrector.Execute(sitk_img)
            
            # Convert back to numpy array
            corrected = sitk.GetArrayFromImage(corrected_img)
            
            return corrected
            
        except Exception as e:
            print(f"Warning: N4 bias correction failed: {e}. Returning original image.")
            return self._normalize_image(image)
    
    def sato_vesselness(
        self,
        image: np.ndarray,
        black_ridges: bool = False
    ) -> np.ndarray:
        """
        Apply Sato vesselness filter to enhance tubular structures (vessels).
        
        The Sato filter computes a measure of vesselness by analyzing the
        eigenvalues of the Hessian matrix at multiple scales. It's particularly
        effective for enhancing blood vessels in angiography images.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            black_ridges: If True, detects dark vessels on bright background
            
        Returns:
            Vesselness-enhanced image (higher values = more vessel-like)
        """
        try:
            # Normalize image to [0, 1]
            img_norm = self._normalize_image(image)
            
            # Apply Sato vesselness filter
            # sigmas: range of vessel widths to detect
            sato_filtered = sato(
                img_norm,
                sigmas=range(
                    self.vesselness_scale_range[0],
                    self.vesselness_scale_range[1],
                    self.vesselness_scale_step
                ),
                black_ridges=black_ridges
            )
            
            return sato_filtered
            
        except Exception as e:
            print(f"Warning: Sato vesselness filter failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def frangi_vesselness(
        self,
        image: np.ndarray,
        black_ridges: bool = False
    ) -> np.ndarray:
        """
        Apply Frangi vesselness filter (alternative to Sato).
        
        The Frangi filter is another Hessian-based method for vessel enhancement.
        It uses different weighting criteria compared to Sato, which may work
        better for certain image types.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            black_ridges: If True, detects dark vessels on bright background
            
        Returns:
            Vesselness-enhanced image (higher values = more vessel-like)
        """
        try:
            # Normalize image to [0, 1]
            img_norm = self._normalize_image(image)
            
            # Apply Frangi vesselness filter
            frangi_filtered = frangi(
                img_norm,
                sigmas=range(
                    self.vesselness_scale_range[0],
                    self.vesselness_scale_range[1],
                    self.vesselness_scale_step
                ),
                black_ridges=black_ridges
            )
            
            return frangi_filtered
            
        except Exception as e:
            print(f"Warning: Frangi vesselness filter failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def otsu_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply Otsu's automatic thresholding for binary segmentation.
        
        Otsu's method automatically determines an optimal threshold value
        by maximizing the between-class variance. Useful for separating
        foreground (brain tissue) from background.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Tuple of (binary_mask, threshold_value)
            - binary_mask: Boolean array where True = foreground
            - threshold_value: The computed threshold
        """
        try:
            # Convert to uint8 for Otsu thresholding
            img_uint8 = self._normalize_to_uint8(image)
            
            # Compute Otsu threshold
            threshold_value = filters.threshold_otsu(img_uint8)
            
            # Create binary mask
            binary_mask = img_uint8 > threshold_value
            
            return binary_mask.astype(np.float32), threshold_value
            
        except Exception as e:
            print(f"Warning: Otsu thresholding failed: {e}. Returning zeros.")
            return np.zeros_like(image, dtype=np.float32), 0.0
    
    def preprocess(self, image: np.ndarray) -> dict:
        """
        Apply complete MRI/MRA preprocessing pipeline.
        
        Args:
            image: Input MRI/MRA image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Dictionary containing:
                - 'preprocessed': Final preprocessed image
                - 'n4_corrected': N4 bias-corrected image (if applied)
                - 'vesselness': Vesselness-enhanced image (if applied)
                - 'otsu_mask': Binary segmentation mask (if applied)
                - 'threshold_value': Otsu threshold value (if applied)
        """
        result = {'original': image}
        current_image = image
        
        # Step 1: N4 Bias Field Correction
        if self.apply_n4:
            current_image = self.n4_bias_correction(current_image)
            result['n4_corrected'] = current_image
        
        # Step 2: Sato Vesselness Filter
        if self.apply_vesselness:
            vesselness_image = self.sato_vesselness(current_image)
            result['vesselness'] = vesselness_image
            current_image = vesselness_image
        
        # Step 3: Otsu Thresholding (optional)
        if self.apply_otsu:
            otsu_mask, threshold_val = self.otsu_threshold(current_image)
            result['otsu_mask'] = otsu_mask
            result['threshold_value'] = threshold_val
            current_image = current_image * otsu_mask
        
        result['preprocessed'] = current_image
        return result
    
    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            return (image - img_min) / (img_max - img_min)
        return image
    
    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 255] uint8 range."""
        normalized = MRIPreprocessor._normalize_image(image)
        return (normalized * 255).astype(np.uint8)


class CTAPreprocessor:
    """
    Preprocessing pipeline for CTA (CT Angiography) images.
    
    Pipeline:
        1. HU Windowing - Focuses on specific tissue density range
        2. CLAHE - Enhances local contrast
        3. Sato Vesselness Filter - Enhances blood vessels
    """
    
    def __init__(
        self,
        apply_hu_windowing: bool = True,
        apply_clahe: bool = True,
        apply_vesselness: bool = True,
        window_center: float = 40.0,
        window_width: float = 80.0,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        vesselness_scale_range: Tuple[int, int] = (1, 8),
        vesselness_scale_step: int = 2
    ):
        """
        Initialize CTA preprocessor.
        
        Args:
            apply_hu_windowing: Whether to apply HU windowing
            apply_clahe: Whether to apply CLAHE contrast enhancement
            apply_vesselness: Whether to apply Sato vesselness filter
            window_center: Center of HU window (default 40 for brain/vessels)
            window_width: Width of HU window (default 80 for brain/vessels)
            clahe_clip_limit: Contrast limit for CLAHE (prevents over-amplification)
            clahe_tile_grid_size: Grid size for CLAHE (8x8 is standard)
            vesselness_scale_range: Scale range for vesselness filter
            vesselness_scale_step: Step size for scale range
        """
        self.apply_hu_windowing = apply_hu_windowing
        self.apply_clahe = apply_clahe
        self.apply_vesselness = apply_vesselness
        self.window_center = window_center
        self.window_width = window_width
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.vesselness_scale_range = vesselness_scale_range
        self.vesselness_scale_step = vesselness_scale_step
    
    def hu_windowing(
        self,
        image: np.ndarray,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Hounsfield Unit (HU) windowing to focus on specific tissue types.
        
        HU windowing selects a specific range of CT intensities (Hounsfield Units)
        to display. Different tissues have different HU values:
        - Air: -1000 HU
        - Water: 0 HU
        - Blood: 30-45 HU
        - Bone: 400+ HU
        
        For brain/vessel imaging, typical settings are:
        - Center: 40 HU (brain tissue and blood)
        - Width: 80 HU (range: 0-80 HU)
        
        Args:
            image: Input CT image in Hounsfield Units
            window_center: Center of window (overrides default if provided)
            window_width: Width of window (overrides default if provided)
            
        Returns:
            Windowed image normalized to [0, 1]
        """
        try:
            # Use provided values or defaults
            center = window_center if window_center is not None else self.window_center
            width = window_width if window_width is not None else self.window_width
            
            # Calculate window bounds
            window_min = center - width / 2
            window_max = center + width / 2
            
            # Apply windowing: clip values outside window
            windowed = np.clip(image, window_min, window_max)
            
            # Normalize to [0, 1] range
            windowed = (windowed - window_min) / (window_max - window_min)
            
            return windowed
            
        except Exception as e:
            print(f"Warning: HU windowing failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def clahe_enhancement(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        tile_grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE enhances local contrast by:
        1. Dividing the image into small tiles (e.g., 8x8 grid)
        2. Applying histogram equalization to each tile independently
        3. Limiting contrast amplification to prevent noise enhancement
        4. Blending tile boundaries for smooth transitions
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            clip_limit: Contrast limit (higher = more aggressive enhancement)
            tile_grid_size: Size of grid for local equalization
            
        Returns:
            Contrast-enhanced image normalized to [0, 1]
        """
        try:
            # Use provided values or defaults
            clip = clip_limit if clip_limit is not None else self.clahe_clip_limit
            grid = tile_grid_size if tile_grid_size is not None else self.clahe_tile_grid_size
            
            # Convert to uint8 for OpenCV CLAHE
            img_uint8 = self._normalize_to_uint8(image)
            
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            
            # Apply CLAHE
            enhanced = clahe.apply(img_uint8)
            
            # Convert back to float [0, 1]
            return enhanced.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"Warning: CLAHE enhancement failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def sato_vesselness(
        self,
        image: np.ndarray,
        black_ridges: bool = False
    ) -> np.ndarray:
        """
        Apply Sato vesselness filter to enhance blood vessels.
        
        Same as MRI vesselness, but applied to CTA images after HU windowing.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            black_ridges: If True, detects dark vessels on bright background
            
        Returns:
            Vesselness-enhanced image (higher values = more vessel-like)
        """
        try:
            # Normalize image to [0, 1]
            img_norm = self._normalize_image(image)
            
            # Apply Sato vesselness filter
            sato_filtered = sato(
                img_norm,
                sigmas=range(
                    self.vesselness_scale_range[0],
                    self.vesselness_scale_range[1],
                    self.vesselness_scale_step
                ),
                black_ridges=black_ridges
            )
            
            return sato_filtered
            
        except Exception as e:
            print(f"Warning: Sato vesselness filter failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def frangi_vesselness(
        self,
        image: np.ndarray,
        black_ridges: bool = False
    ) -> np.ndarray:
        """
        Apply Frangi vesselness filter (alternative to Sato).
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            black_ridges: If True, detects dark vessels on bright background
            
        Returns:
            Vesselness-enhanced image (higher values = more vessel-like)
        """
        try:
            # Normalize image to [0, 1]
            img_norm = self._normalize_image(image)
            
            # Apply Frangi vesselness filter
            frangi_filtered = frangi(
                img_norm,
                sigmas=range(
                    self.vesselness_scale_range[0],
                    self.vesselness_scale_range[1],
                    self.vesselness_scale_step
                ),
                black_ridges=black_ridges
            )
            
            return frangi_filtered
            
        except Exception as e:
            print(f"Warning: Frangi vesselness filter failed: {e}. Returning normalized image.")
            return self._normalize_image(image)
    
    def preprocess(self, image: np.ndarray) -> dict:
        """
        Apply complete CTA preprocessing pipeline.
        
        Args:
            image: Input CTA image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Dictionary containing:
                - 'preprocessed': Final preprocessed image
                - 'hu_windowed': HU windowed image (if applied)
                - 'clahe_enhanced': CLAHE enhanced image (if applied)
                - 'vesselness': Vesselness-enhanced image (if applied)
        """
        result = {'original': image}
        current_image = image
        
        # Step 1: HU Windowing
        if self.apply_hu_windowing:
            current_image = self.hu_windowing(current_image)
            result['hu_windowed'] = current_image
        
        # Step 2: CLAHE Enhancement (optional, can be applied after HU windowing)
        if self.apply_clahe:
            clahe_image = self.clahe_enhancement(current_image)
            result['clahe_enhanced'] = clahe_image
            # Note: You can choose to continue with CLAHE or HU windowed
            # For vessel detection, HU windowing alone often works better
            # current_image = clahe_image
        
        # Step 3: Sato Vesselness Filter
        if self.apply_vesselness:
            vesselness_image = self.sato_vesselness(current_image)
            result['vesselness'] = vesselness_image
            current_image = vesselness_image
        
        result['preprocessed'] = current_image
        return result
    
    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            return (image - img_min) / (img_max - img_min)
        return image
    
    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 255] uint8 range."""
        normalized = CTAPreprocessor._normalize_image(image)
        return (normalized * 255).astype(np.uint8)


def create_preprocessor(modality: str, **kwargs) -> Optional[object]:
    """
    Factory function to create appropriate preprocessor based on modality.
    
    Args:
        modality: Imaging modality ('MRI', 'MRA', 'MRI T1post', 'MRI T2', 'CTA', etc.)
        **kwargs: Additional arguments passed to preprocessor constructor
        
    Returns:
        MRIPreprocessor for MRI/MRA modalities, CTAPreprocessor for CTA
        
    Examples:
        >>> # Create MRI preprocessor with default settings
        >>> mri_prep = create_preprocessor('MRI')
        >>> result = mri_prep.preprocess(image)
        >>> preprocessed_img = result['preprocessed']
        
        >>> # Create CTA preprocessor with custom HU window
        >>> cta_prep = create_preprocessor('CTA', window_center=50, window_width=100)
        >>> result = cta_prep.preprocess(image)
        >>> preprocessed_img = result['preprocessed']
    """
    modality_upper = modality.upper()
    
    if 'MRI' in modality_upper or 'MRA' in modality_upper:
        return MRIPreprocessor(**kwargs)
    elif 'CTA' in modality_upper or 'CT' in modality_upper:
        return CTAPreprocessor(**kwargs)
    else:
        print(f"Warning: Unknown modality '{modality}'. No preprocessor created.")
        return None


# Example usage and testing
if __name__ == "__main__":
    print("Medical Image Preprocessing Module")
    print("=" * 60)
    print("\nAvailable preprocessing pipelines:")
    print("\n1. MRI/MRA Pipeline:")
    print("   - N4 Bias Field Correction: Removes intensity non-uniformity")
    print("   - Sato Vesselness Filter: Enhances blood vessels")
    print("   - Otsu Thresholding: Automatic segmentation (optional)")
    print("\n2. CTA Pipeline:")
    print("   - HU Windowing: Focuses on brain/vessel HU range (40±40)")
    print("   - CLAHE: Enhances local contrast (optional)")
    print("   - Sato Vesselness Filter: Enhances blood vessels")
    print("\n" + "=" * 60)
    
    # Create sample preprocessors
    print("\nExample: Creating preprocessors...")
    
    # MRI preprocessor with default settings
    mri_prep = create_preprocessor('MRI')
    print(f"✓ MRI Preprocessor created: {type(mri_prep).__name__}")
    
    # CTA preprocessor with custom settings
    cta_prep = create_preprocessor(
        'CTA',
        window_center=40,
        window_width=80,
        apply_clahe=True,
        apply_vesselness=True
    )
    print(f"✓ CTA Preprocessor created: {type(cta_prep).__name__}")
    
    print("\nTo use in your pipeline:")
    print("  from data.preprocessing import create_preprocessor")
    print("  preprocessor = create_preprocessor('MRI')")
    print("  result = preprocessor.preprocess(image)")
    print("  preprocessed_image = result['preprocessed']")
