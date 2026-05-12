import os
import argparse
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def dicom_series_to_nifti(input_dir, output_path):
    """
    Reads a DICOM series from a directory and saves it as a single NIfTI file.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    
    if not dicom_names:
        return False
        
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    sitk.WriteImage(image, output_path)
    return True

def main(args):
    csv_path = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find train.csv at {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Load error UIDs to skip
    error_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "error_data.yaml")
    error_uids = set()
    if os.path.exists(error_yaml):
        with open(error_yaml, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('- '):
                    uid = line[2:].split('#')[0].strip()
                    if uid:
                        error_uids.add(uid)
    print(f"Loaded {len(error_uids)} error UIDs to skip.")
    
    # Create the output directory
    output_dir = os.path.join(args.data_dir, "series_nifti")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting conversion. Output will be saved to: {output_dir}")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # Iterate with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting DICOMs"):
        sid = str(row['SeriesInstanceUID'])
        
        if sid in error_uids:
            skipped_count += 1
            continue
            
        image_dir = os.path.join(args.data_dir, "series", sid)
        output_path = os.path.join(output_dir, f"{sid}.nii.gz")
        
        if os.path.exists(output_path):
            success_count += 1
            continue # Already converted
            
        if os.path.exists(image_dir):
            try:
                success = dicom_series_to_nifti(image_dir, output_path)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"\nError converting {sid}: {e}")
                fail_count += 1
        else:
            fail_count += 1
            
    print("\nConversion Completed!")
    print(f"Successfully converted: {success_count}")
    print(f"Failed or missing: {fail_count}")
    print(f"Skipped (corrupted): {skipped_count}")
    print("\nIf you use the converted files, remember to update dataset.py to point to 'series_nifti' instead of 'series' and load them as single files!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM series to NIfTI")
    parser.add_argument("--data_dir", type=str, default="G:\\FYP_Data\\rsna-intracranial-aneurysm-detection", help="Path to the dataset")
    args = parser.parse_args()
    main(args)
