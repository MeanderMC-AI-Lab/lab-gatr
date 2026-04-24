from argparse import ArgumentParser
from pathlib import Path
import torch
import os
import json
import numpy as np
import pandas as pd
import open3d as o3d
import torch_geometric as pyg
from torch_geometric.data import Data
from lab_gatr import LaBGATr
from lab_gatr.transforms import PointCloudPoolingScales
from gatr.interface import embed_point, embed_oriented_plane, extract_oriented_plane
from scipy.spatial import cKDTree
# from datasets import cartesian_to_positional_encoding
from datasets import angle_weighted_normals
from losses import ChamferLoss

parser = ArgumentParser()
parser.add_argument('--data_root', type=str, default='/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/pointclouds')
parser.add_argument('--model_weights', type=str)
parser.add_argument('--rel_sampling_ratio', type=float, default=0.05)
parser.add_argument('--interp_simplex', type=str, choices=['triangle', 'tetrahedron'], default='tetrahedron')
parser.add_argument('--pooling_mode', type=str, choices=['message_passing', 'cross_attention'], default='cross_attention')
parser.add_argument('--d_model', type=int, default=8)
parser.add_argument('--num_blocks', type=int, default=10)
parser.add_argument('--num_attn_heads', type=int, default=4)
parser.add_argument('--eval_thr', type=float, default=None)
parser.add_argument('--fold', type=int, default=None)
args = parser.parse_args()

def orient_normals_negative_zaxis(normals):
    # normals = np.asarray(normals)
    mask = normals[:, 2] > 0
    normals[mask] *= -1
    return normals
    # return o3d.cuda.pybind.utility.Vector3dVector(normals)


def rmse_fn(pts_a, pts_b):
    return np.linalg.norm(pts_a - pts_b, axis=-1)


def mae_fn(pts_a, pts_b):
    return np.sum(np.abs(pts_a - pts_b), axis=-1) / 3.


def magn_fn(pts_a, pts_b):
    magn_a = np.linalg.norm(pts_a, axis=-1).clip(min=1e-8)
    magn_b = np.linalg.norm(pts_b, axis=-1)
    return np.abs(magn_a - magn_b) / magn_a * 100


def ang_fn(pts_a, pts_b):
    dot = np.sum(pts_b * pts_a, axis=-1)
    denom = np.linalg.norm(pts_b, axis=-1).clip(min=1e-8) * np.linalg.norm(pts_a, axis=-1).clip(min=1e-8)
    cos = np.clip(dot / denom, a_min=-1., a_max=1.)
    return np.arccos(cos) * 180. / np.pi


class GeometricAlgebraInterface:
    num_input_channels = 2
    num_input_scalars = 1  # 36
    num_output_channels = 1
    num_output_scalars = None

    @staticmethod
    @torch.no_grad()
    def embed(data):
        multivectors = torch.cat((
            embed_point(data.pos).view(-1, 1, 16),
            # embed_oriented_plane(data.x[:, :3], data.pos).view(-1, 1, 16)
            embed_oriented_plane(data.norm, data.pos).view(-1, 1, 16)
        ), dim=1)
        # scalars = data.x[:, 3:]
        scalars = torch.zeros(data.pos.shape[0], 1, device=data.pos.device)  # scalars cannot be None, so inputting zeros instead
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_oriented_plane(multivectors).squeeze()
        return output


def positional_encoding(data):
    # data.x = torch.cat([data.norm, data.pos_enc], dim=1)
    data.x = torch.cat([data.norm], dim=1)
    return data


def main():
    transform = pyg.transforms.Compose((
        PointCloudPoolingScales(
            rel_sampling_ratios = (args.rel_sampling_ratio,),
            interp_simplex = args.interp_simplex
        ),
        # positional_encoding
    ))
    
    device = torch.device("cuda")
    
    neural_network = LaBGATr(
        GeometricAlgebraInterface,
        d_model = args.d_model,
        num_blocks = args.num_blocks,
        num_attn_heads = args.num_attn_heads,
        pooling_mode = args.pooling_mode
    )
    neural_network.to(device)
    # , map_location=torch.device("cpu")
    neural_network.load_state_dict(torch.load(args.model_weights, weights_only=True))
    neural_network.eval()

    chamfer_loss = ChamferLoss()

    ct_dirs = sorted([d for d in os.listdir(args.data_root) if d.endswith("CT+")])
    if args.fold is not None:
        with open("folds.json", "r") as f:
            test_idxs = json.load(f)["10-fold"][str(args.fold)]
        patient_idxs = [int(d.split("_")[1]) for d in ct_dirs]
        ct_dirs = [d for d, pt in zip(ct_dirs, patient_idxs) if pt in test_idxs]
    print("Selected files:", ct_dirs)

    results = []
    results_json = {}
    with torch.no_grad():
        for dir in ct_dirs:
            print(f"Processing {dir}:")
            
            # Load CT data
            data_dir = os.path.join(args.data_root, dir)
            
            # ct_mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, "ct_abd_segm_mesh.ply"))
            # vertices = torch.as_tensor(np.asarray(ct_mesh.vertices)).float() * 100.
            # faces = torch.as_tensor(np.asarray(ct_mesh.triangles)).int()
            # normals = torch.as_tensor(orient_normals_negative_zaxis(np.asarray(ct_mesh.vertex_normals))).float(),
            # normals = angle_weighted_normals(vertices, faces)

            ct_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, "ct_abd_segm_pcd_clean.ply"))
            vertices = torch.as_tensor(
                np.asarray(ct_pcd.points)
            ).float() * 100.
            normals = torch.as_tensor(
                orient_normals_negative_zaxis(np.asarray(ct_pcd.normals))
            ).float()
            print(vertices.shape, normals.shape)
            
            data = Data(
                pos = vertices,
                norm = normals
            )
            data.to(device)
            data = transform(data)
            
            # Make prediction
            prediction = neural_network(data)
            pred = data.pos + prediction

            # Store result
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred.cpu().numpy() / 100.)
            o3d.io.write_point_cloud(os.path.join(data_dir, "ct_pred_pcd.ply"), pcd)

            # Evaluation
            dpt_start = o3d.io.read_point_cloud(os.path.join(data_dir, "depth_pcd_start.ply"))
            dpt_start_pts = torch.as_tensor(np.asarray(dpt_start.points), device=device).float() * 100
            dpt_end = o3d.io.read_point_cloud(os.path.join(data_dir, "depth_pcd_end.ply"))
            dpt_end_pts = torch.as_tensor(np.asarray(dpt_end.points), device=device).float() * 100
            anns = torch.load(os.path.join(data_dir, "annotations.pt"), weights_only=False)
            dpt_anns_start = anns["annotations_start"] * 100.
            dpt_anns_end = anns["annotations_end"] * 100.
            
            cd_start = chamfer_loss(data.pos.unsqueeze(dim=0),
                                    dpt_start_pts.unsqueeze(dim=0)).cpu().item()
            cd_end = chamfer_loss(pred.unsqueeze(dim=0),
                                  dpt_end_pts.unsqueeze(dim=0)).cpu().item()
            print(f"> Chamfer: {cd_start:.2f} (before), {cd_end:.2f} (after)")

            ct_start_pts = data.pos.cpu().numpy()
            tree = cKDTree(ct_start_pts)
            deltas, idxs = tree.query(dpt_anns_start)
            orig_num_anns = len(deltas)
            if args.eval_thr is not None:
                mask = deltas <= args.eval_thr
                dpt_anns_start = dpt_anns_start[mask]
                dpt_anns_end = dpt_anns_end[mask]
                idxs = idxs[mask]
                deltas = deltas[mask]

            dpt_anns_delta = dpt_anns_end - dpt_anns_start
            ct_anns_start = ct_start_pts[idxs]
            ct_anns_delta = prediction[idxs].cpu().numpy()
            mean_delta = np.mean(deltas)
            num_anns_removed = orig_num_anns - len(deltas)
            print(f"> Mean delta (depth -> CT anns): {mean_delta:.2f}, Num anns removed: {num_anns_removed} (thr={args.eval_thr})")

            rmse = rmse_fn(ct_anns_delta, dpt_anns_delta)
            mae = mae_fn(ct_anns_delta, dpt_anns_delta)
            magn = magn_fn(ct_anns_delta, dpt_anns_delta)
            ang = ang_fn(ct_anns_delta, dpt_anns_delta)
            rmse_mean = np.mean(rmse)
            mae_mean = np.mean(mae)
            magn_mean = np.mean(magn)
            ang_mean = np.mean(ang)
            print(f"> RMSE: {rmse_mean:.2f}, MAE: {mae_mean:.2f}, MAGN: {magn_mean:.2f}, ANG: {ang_mean:.2f}")

            results_dict = {
                "patient": dir,
                "cd_start": cd_start,
                "cd_end": cd_end,
                "cd_anns": mean_delta,
                "anns_removed": num_anns_removed,
                "rmse": rmse_mean,
                "mae": mae_mean,
                "magn": magn_mean,
                "ang": ang_mean
            }
            results.append(results_dict)
            results_dict["landmarks"] = {
                "anns_displ": dpt_anns_delta.tolist(),
                "pred_displ": ct_anns_delta.tolist(),
                "anns_start": dpt_anns_start.tolist(),
                "anns_end": dpt_anns_end.tolist(),
                "pred_pos": pred[idxs].cpu().numpy().tolist(),
                "rmse": rmse.tolist(),
                "mae": mae.tolist(),
                "magn": magn.tolist(),
                "ang": ang.tolist()
            }
            results_json[dir] = results_dict

    new_results = pd.DataFrame(results)
    results_csv = os.path.join(args.data_root, "lab-gatr_results.csv")
    if os.path.isfile(results_csv):
        earlier_results = pd.read_csv(results_csv)
        all_results = pd.concat([earlier_results, new_results], ignore_index=True)
        all_results.to_csv(results_csv, index=False)
    else:
        new_results.to_csv(results_csv, index=False)

    path_json = os.path.join(args.data_root, "lab-gatr_results.json")
    if os.path.isfile(path_json):
        with open(path_json, "r") as f:
            earlier_results = json.load(f)
        results_json = earlier_results | results_json
    with open(path_json, "w") as f:
        json.dump(results_json, f)

if __name__ == '__main__':
    main()

