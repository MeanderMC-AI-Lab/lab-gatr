import os
import copy
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint

ROOT = "/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds_iso"
DEPTHS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/camera_angles.csv"
DEPTHS = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/raw"
SKIP = ["24_05_CT+"]


def transform_camera(depth_df, coords):
    coords = np.asarray(coords)
    if depth_df["swap_axis"].item():
        coords[:, [0, 1]] = coords[:, [1, 0]]
    if depth_df["reflect_axis"].item():
        coords[:, 1] = -coords[:, 1]
    if depth_df["swap_axis"].item() and depth_df["reflect_axis"].item():
        coords[:, 0] = -coords[:, 0]
    return coords


def transform_o3d(coords, T, return_pcd=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.transform(T)
    if return_pcd:
        return pcd
    return np.asarray(pcd.points)


def register_depth_to_ct(pcd_dirs):
    depths = pd.read_csv(DEPTHS_CSV)
    icp_results = []
    for dir in pcd_dirs:
        path = os.path.join(ROOT, dir)
        ct_pcd = o3d.io.read_point_cloud(os.path.join(path, "ct_segm_pcd_centered.ply"))
        depth_pcd = o3d.io.read_point_cloud(os.path.join(path, "depth_pcd.ply"))
        
        # Register depth pointcloud to ct pointcloud
        result = registration_icp(
            source = depth_pcd,
            target = ct_pcd,
            max_correspondence_distance = 0.01,
            init = np.eye(4),
            estimation_method = TransformationEstimationPointToPoint(),
        )
        icp_results.append({
            "patient": dir,
            "fitness": result.fitness,
            "inlier_rmse": result.inlier_rmse,
            "len_ct": len(ct_pcd.points),
            "len_depth": len(depth_pcd.points)
        })
        print(f"> {dir}: {result.fitness=:.3f}, {result.inlier_rmse=:.3f}, {len(ct_pcd.points)=}, {len(depth_pcd.points)=}")
    
        # Transform depth pointcloud and store as ply
        T = result.transformation
        depth_pcd.transform(T)
        o3d.io.write_point_cloud(os.path.join(path, "depth_pcd_start.ply"), depth_pcd)
    
        # Load inflated depth pointcloud and annotations, transform and store
        depth_file = f"{dir}.pt"
        depth = depths[depths["patient"]==depth_file]
        depth_data = torch.load(os.path.join(DEPTHS, depth_file), weights_only=False)
        
        depth_pcd_end = transform_o3d(
            transform_camera(depth, depth_data["target_points"]),
            T)
        o3d.io.write_point_cloud(os.path.join(path, "depth_pcd_end.ply"), depth_pcd_end)
        
        anns_start = transform_o3d(
            transform_camera(depth, depth_data["annotations_start"]),
            T, return_pcd=False)
        anns_end = transform_o3d(
            transform_camera(depth, depth_data["annotations_end"]),
            T, return_pcd=False)
        torch.save({
            "annotations_start": anns_start,
            "annotations_end": anns_end
        }, os.path.join(path, "annotations.pt"))
    
        # Store the transform for possible later use
        np.save(os.path.join(path, "icp_transform.npy"), T)
    
    results_df = pd.DataFrame(icp_results)
    results_df.to_csv(os.path.join(ROOT, "icp_results.csv"), index=False)


def segment_abdominal_area(pcd_dirs):
    for dir in pcd_dirs:
        path = os.path.join(ROOT, dir)
        ct_pcd = o3d.io.read_point_cloud(os.path.join(path, "ct_segm_pcd_centered.ply"))
        depth_pcd = o3d.io.read_point_cloud(os.path.join(path, "depth_pcd_start.ply"))

        ct_pts = np.asarray(ct_pcd.points)
        depth_pts = np.asarray(depth_pcd.points)

        # Get x and y boundaries of depth pointcloud and create mask for ct pointcloud
        xmin, ymin = depth_pts[:, :2].min(axis=0)
        xmax, ymax = depth_pts[:, :2].max(axis=0)
        zmax = depth_pts[:, 2].max()
        print(f"> {dir}: {xmin=:.3f}, {xmax=:.3f}, {ymin=:.3f}, {ymax=:.3f}, {zmax=:.3f}")
        mask = (
            (ct_pts[:, 0] >= xmin) & (ct_pts[:, 0] <= xmax) &
            (ct_pts[:, 1] >= ymin) & (ct_pts[:, 1] <= ymax) &
            (ct_pts[:, 2] <= zmax)
        )

        # Create new pointcloud with masked points from ct
        ct_segm_pcd = o3d.geometry.PointCloud()
        ct_segm_pcd.points = o3d.utility.Vector3dVector(ct_pts[mask])
        o3d.io.write_point_cloud(os.path.join(path, "ct_abd_segm_pcd.ply"), ct_segm_pcd)


def simplify_mesh(pcd_dirs):
    for dir in tqdm(pcd_dirs):
        path = os.path.join(ROOT, dir)
        mesh = o3d.io.read_triangle_mesh(os.path.join(path, "ct_abd_segm_mesh.ply"))

        # Resample mesh
        pcd = mesh.sample_points_poisson_disk(number_of_points=50000)
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=2.0
        )
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=8,
            radius=0.02
        )
        
        # Find largest cluster of triangles and only keep this one
        # labels, num_triangles, _ = mesh.cluster_connected_triangles()
        # labels = np.asarray(labels)
        # num_triangles = np.asarray(num_triangles)
        # largest = int(np.argmax(num_triangles))
        # mask = (labels == largest)
        # mesh.remove_triangles_by_mask(~mask)
        # mesh.remove_unreferenced_vertices()
        # mesh.compute_vertex_normals()

        # Downsample mesh
        # TODO: implement

        # Clean-up mesh
        # mesh = mesh.remove_degenerate_triangles()
        # mesh = mesh.remove_duplicated_triangles()
        # mesh = mesh.remove_duplicated_vertices()
        # mesh = mesh.remove_non_manifold_edges()
        # mesh.remove_unreferenced_vertices()
        # mesh.compute_vertex_normals()

        # Store new mesh
        # o3d.io.write_triangle_mesh(os.path.join(path, "ct_abd_segm_mesh_clean.ply"), mesh)
        o3d.io.write_point_cloud(os.path.join(path, "ct_abd_segm_pcd_clean.ply"), pcd)
        

if __name__ == "__main__":
    pcd_dirs = [d for d in os.listdir(ROOT) if d.endswith("CT+")]
    pcd_dirs = sorted(pcd_dirs)
    pcd_dirs = [d for d in pcd_dirs if d not in SKIP]
    
    # register_depth_to_ct(pcd_dirs)
    # segment_abdominal_area(pcd_dirs)
    simplify_mesh(pcd_dirs)

