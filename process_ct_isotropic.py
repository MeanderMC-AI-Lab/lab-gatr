import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm


PATIENTS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/body_composition.csv"
DICOMS = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/dicom"
SKIP = ["24_05_CT+", "24_06_CT+", "24_07_CT+"]


if __name__ == "__main__":
    patients = pd.read_csv(PATIENTS_CSV)

    for pt_idx, pt in tqdm(patients.iterrows(), total=len(patients)):
        pt_key = pt["Key"]
        pt_studyid = str(pt["PatientID"])

        if pt_key in SKIP:
            print(f"Skipping {pt_key} ({pt_studyid})")
            continue

        pt_dir = os.path.join(DICOMS, pt_studyid)
        ct_dir = [d for d in os.listdir(pt_dir) if d.startswith("302")][0]
        ct_dir = os.path.join(pt_dir, ct_dir)

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(ct_dir)
        series_files = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
        reader.SetFileNames(series_files)
        ct = reader.Execute()

        out_spacing = np.array([1., 1., 1.], dtype=float)
        in_spacing = np.array(ct.GetSpacing(), dtype=float)
        in_size = np.array(ct.GetSize(), dtype=int)
        out_size = np.round(in_size * (in_spacing / out_spacing)).astype(int)
        out_size = [int(x) for x in out_size]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(tuple(out_spacing))
        resampler.SetSize(out_size)
        resampler.SetOutputDirection(ct.GetDirection())
        resampler.SetOutputOrigin(ct.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        ct_iso = resampler.Execute(ct)

        print(f"Created isotropic CT with size {ct_iso.GetSize()} and spacing {ct_iso.GetSpacing()}")
        sitk.WriteImage(ct_iso, os.path.join(pt_dir, "ct_iso.nii.gz"))


