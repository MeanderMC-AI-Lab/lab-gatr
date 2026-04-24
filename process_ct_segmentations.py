import os
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
import open3d as o3d
# import plotly.graph_objects as go

PATIENTS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/body_composition.csv"
SEGMENTATIONS = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/dicom"  #segmentations
DEPTHS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/camera_angles.csv"
DEPTHS = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/raw"
WRITE_DIR = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds_iso"  #pointclouds
SKIP = ["24_05_CT+"]


def segment_abdomen(data, affine):
    subcutaneous_fat = (data == 1)
    idxs = np.arange(data.shape[1])  # Assert orientation is RAS
    idxs = idxs[None, :, None]
    top_mask = np.where(subcutaneous_fat, idxs, -1).max(axis=1)
    x, z = np.where(top_mask >= 0)
    y = top_mask[x, z]
    top_coords = np.column_stack([x, y, z])
    return apply_affine(affine, top_coords)


def recenter_pointcloud(points):
    center = np.median(points, axis=0)
    return points - center


def convert_to_meters(points):
    return points / 1000


def lsp_orientation(coords):
    R = coords[:, 0]
    A = coords[:, 1]
    S = coords[:, 2]
    return np.stack([-R, S, -A], axis=1)


if __name__ == "__main__":
    patients = pd.read_csv(PATIENTS_CSV)
    depths = pd.read_csv(DEPTHS_CSV)
    
    for pt_idx, pt in tqdm(patients.iterrows(), total=len(patients)):
        
        # Find CT and put into RAS-coordinates
        pt_key = pt["Key"]
        pt_studyid = pt["PatientID"]
        if pt_key in SKIP:
            print(f"Skipping {pt_key} ({pt_studyid})")
            continue
        pt_write_dir = os.path.join(WRITE_DIR, pt_key)
        os.makedirs(pt_write_dir, exist_ok=True)
        ct_folder = os.path.join(SEGMENTATIONS, str(pt_studyid))
        # ct_file = glob(os.path.join(ct_folder, "*.nii"))[0]
        ct_file = os.path.join(ct_folder, "segmentations", "subcutaneous_fat.nii.gz")
        ct_img = nib.load(ct_file)
        ct_img = nib.as_closest_canonical(ct_img)
        ct_data = ct_img.get_fdata()
        ct_affine = ct_img.affine

        # Segment most anterior area, recenter, convert to meters and LSP-coordinates
        abdominal_coords = segment_abdomen(ct_data, ct_affine)
        abdominal_coords = recenter_pointcloud(abdominal_coords)
        abdominal_coords = convert_to_meters(abdominal_coords)
        abdominal_coords = lsp_orientation(abdominal_coords)

        # Write point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(abdominal_coords)
        o3d.io.write_point_cloud(os.path.join(pt_write_dir, "ct_segm_pcd.ply"), pcd)

        # Write depth scan point cloud as reference
        depth_file = f"{pt_key}.pt"
        depth = depths[depths["patient"]==depth_file]
        depth_data = torch.load(os.path.join(DEPTHS, depth_file), weights_only=False)
        depth_coords = depth_data["input_points"]
    
        if depth["swap_axis"].item():
            depth_coords[:, [0, 1]] = depth_coords[:, [1, 0]]
        if depth["reflect_axis"].item():
            depth_coords[:, 1] = -depth_coords[:, 1]
        if depth["swap_axis"].item() and depth["reflect_axis"].item():
            depth_coords[:, 0] = -depth_coords[:, 0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_coords)
        o3d.io.write_point_cloud(os.path.join(pt_write_dir, "depth_pcd.ply"), pcd)

