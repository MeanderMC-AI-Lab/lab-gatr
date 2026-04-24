import os
import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm


PATIENTS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/body_composition.csv"
NEW_POINTCLOUDS = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds_iso"
OLD_POINTCLOUDS = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds"
SKIP = ["24_05_CT+"]


if __name__ == "__main__":
    patients = pd.read_csv(PATIENTS_CSV)
    
    for pt_idx, pt in tqdm(patients.iterrows(), total=len(patients)):        
        pt_key = pt["Key"]
        pt_studyid = pt["PatientID"]
        if pt_key in SKIP:
            print(f"Skipping {pt_key} ({pt_studyid})")
            continue

        # Find the original translation
        old_pcd = o3d.io.read_point_cloud(
            os.path.join(OLD_POINTCLOUDS, pt_key, "ct_segm_pcd.ply")
        )
        old_pcd_rec = o3d.io.read_point_cloud(
            os.path.join(OLD_POINTCLOUDS, pt_key, "ct_segm_pcd_centered.ply")
        )
        translation = np.subtract(
            np.asarray(old_pcd_rec.points).mean(axis=0),
            np.asarray(old_pcd.points).mean(axis=0)
        )

        # Apply to new pointcloud and store result
        new_pcd = o3d.io.read_point_cloud(
            os.path.join(NEW_POINTCLOUDS, pt_key, "ct_segm_pcd.ply")
        )
        new_pcd.translate(translation)
        o3d.io.write_point_cloud(
            os.path.join(NEW_POINTCLOUDS, pt_key, "ct_segm_pcd_centered.ply"), new_pcd
        )
