import os
import numpy as np
import open3d as o3d


def save_pred_and_gt_pointclouds(save_dir, start, gt, pred, idx):
    save_dir = os.path.join(save_dir, "vis")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"start_{idx:04d}.ply"), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start + gt)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"end_{idx:04d}.ply"), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start + pred)
    o3d.io.write_point_cloud(os.path.join(save_dir, f"pred_{idx:04d}.ply"), pcd)