import os
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


def calc_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.zeros_like(vertices)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = F.normalize(normals, dim=1, eps=1e-12)

    # Orient normals towards the direction of the camera
    mask = normals[:, 2] > 0
    normals[mask] *= -1
    return normals
    

def save_pred_and_gt_pointclouds(save_dir, start, start_norm, end, end_norm, pred, faces, idx):
    save_dir = os.path.join(save_dir, "vis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start.numpy())
    pcd.normals = o3d.utility.Vector3dVector(start_norm.numpy())
    o3d.io.write_point_cloud(os.path.join(save_dir, f"start_{idx:04d}.ply"), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(end.numpy())
    pcd.normals = o3d.utility.Vector3dVector(end_norm.numpy())
    o3d.io.write_point_cloud(os.path.join(save_dir, f"end_{idx:04d}.ply"), pcd)

    pred = start + pred
    pred_norm = calc_normals(pred, faces)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred.numpy())
    pcd.normals = o3d.utility.Vector3dVector(pred_norm.numpy())
    o3d.io.write_point_cloud(os.path.join(save_dir, f"pred_{idx:04d}.ply"), pcd)
    