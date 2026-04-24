import os
import torch
import numpy as np
import pandas as pd
import open3d as o3d


WRITE_DIR = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/raw"
DEPTHS_CSV = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/camera_angles.csv"
DEPTHS = "/data/Predict-Pneumoperitoneum_LaB-GATr/dataset/raw_paper"
SKIP = []
SELECT = []
TRANSFORM = False
VOXEL_SIZE = 0.001


def transform_camera(depth_df, coords):
    coords = np.asarray(coords)
    if depth_df["swap_axis"].item():
        coords[:, [0, 1]] = coords[:, [1, 0]]
    if depth_df["reflect_axis"].item():
        coords[:, 1] = -coords[:, 1]
    if depth_df["swap_axis"].item() and depth_df["reflect_axis"].item():
        coords[:, 0] = -coords[:, 0]
    coords = torch.from_numpy(coords)
    return coords


def resample_mesh(vertices, faces, voxel_size=0.001):
    # num_triangles = 2 * num_vertices  # Rule of thumb
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    
    # mesh_sub = mesh.simplify_quadric_decimation(
        # target_number_of_triangles=num_triangles)
    mesh_sub = mesh.simplify_vertex_clustering(
        voxel_size = voxel_size
    )

    mesh_sub.remove_duplicated_vertices()
    mesh_sub.remove_duplicated_triangles()
    mesh_sub.remove_degenerate_triangles()
    mesh_sub.remove_non_manifold_edges()
    mesh_sub.compute_vertex_normals()

    vertices_new = torch.from_numpy(np.asarray(mesh_sub.vertices))
    faces_new = torch.from_numpy(np.asarray(mesh_sub.triangles))
    normals_new = torch.from_numpy(np.asarray(mesh_sub.vertex_normals))
    return vertices_new, faces_new, normals_new
    

def preprocess_pointclouds(pcd_files, transform=False, resampling=None):
    depths = pd.read_csv(DEPTHS_CSV)
    mesh_sizes = []
    for pcd_file in pcd_files:
        print(f"Processing {pcd_file}")
        depth = depths[depths["patient"]==pcd_file]
        depth_data = torch.load(os.path.join(DEPTHS, pcd_file), weights_only=False)

        if transform:
            for key in ["input_points", "target_points", "annotations_start",
                        "annotations_end", "input_normals", "displacements"]:
                depth_data[key] = transform_camera(depth, depth_data[key])
            print(f" Swap: {depth['swap_axis'].item()}, Reflect: {depth['reflect_axis'].item()}")
        
        if resampling is not None:
            num_vert_input_orig = depth_data["input_points"].shape[0]
            num_facs_input_orig = depth_data["input_faces"].shape[0]
            num_vert_target_orig = depth_data["target_points"].shape[0]
            num_facs_target_orig = depth_data["target_faces"].shape[0]
            # Resample input mesh
            vertices_new, faces_new, normals_new = resample_mesh(
                depth_data["input_points"],
                depth_data["input_faces"],
                voxel_size = resampling
            )
            num_vert_input = vertices_new.shape[0]
            num_facs_input = faces_new.shape[0]
            depth_data["input_points"] = vertices_new
            depth_data["input_faces"] = faces_new
            depth_data["input_normals"] = normals_new
            # Resample output mesh
            vertices_new, faces_new, normals_new = resample_mesh(
                depth_data["target_points"],
                depth_data["target_faces"],
                voxel_size = resampling
            )
            num_vert_target = vertices_new.shape[0]
            num_facs_target = faces_new.shape[0]
            depth_data["target_points"] = vertices_new
            depth_data["target_faces"] = faces_new
            depth_data["target_normals"] = normals_new
            # Log output
            print(f" Input mesh: {num_vert_input} (V), {num_facs_input} (F)")
            print(f" Target mesh: {num_vert_input} (V), {num_facs_input} (F)")
            mesh_sizes.append({
                "orig_in_verts": num_vert_input_orig,
                "orig_in_faces": num_facs_input_orig,
                "orig_tg_verts": num_vert_target_orig,
                "orig_tg_faces": num_facs_target_orig,
                "in_verts": num_vert_input,
                "in_faces": num_facs_input,
                "tg_verts": num_vert_target,
                "tg_faces": num_facs_target
            })
        
        torch.save(depth_data, os.path.join(WRITE_DIR, pcd_file))
    if mesh_sizes:
        sizes = pd.DataFrame(mesh_sizes)
        sizes.to_csv(os.path.join(WRITE_DIR, "mesh_sizes.csv"), index=False)

if __name__ == "__main__":
    pcd_files = [f for f in os.listdir(DEPTHS) if f.endswith(".pt")]
    pcd_files = sorted(pcd_files)
    pcd_files = [f for f in pcd_files if f.replace(".pt", "") not in SKIP]
    if SELECT:
        pcd_files = [f for f in pcd_files if f.replace(".pt", "") in SELECT]
    os.makedirs(WRITE_DIR, exist_ok=True)
    preprocess_pointclouds(pcd_files, transform=TRANSFORM, resampling=VOXEL_SIZE)

