from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import open3d as o3d
import torch_geometric as pyg
from torch_geometric.data import Data
from lab_gatr import LaBGATr
from lab_gatr.transforms import PointCloudPoolingScales
from gatr.interface import embed_point, embed_oriented_plane, extract_oriented_plane
from datasets import cartesian_to_positional_encoding


def calculate_inputs():
    multivectors = 0
    scalars = 0
    if args.feat_norm:
        multivectors += 1
    if args.feat_umbilicus:
        multivectors += 1
        scalars += 1
    if args.feat_patient:
        scalars += 5
    if args.pos_enc:
        scalars += 36
    return multivectors, scalars


parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default='/data/Predict-Pneumoperitoneum_LaB-GATr/ct_scans/24_12_CT+/ct_scan.pt')
parser.add_argument('--model_weights', type=str)
parser.add_argument('--rel_sampling_ratio', type=float, default=0.05)
parser.add_argument('--interp_simplex', type=str, choices=['triangle', 'tetrahedron'], default='tetrahedron')
parser.add_argument('--pooling_mode', type=str, choices=['message_passing', 'cross_attention'], default='cross_attention')
parser.add_argument('--d_model', type=int, default=8)
parser.add_argument('--num_blocks', type=int, default=10)
parser.add_argument('--num_attn_heads', type=int, default=4)
args = parser.parse_args()


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
            embed_oriented_plane(data.x[:, :3], data.pos).view(-1, 1, 16)
        ), dim=1)
        # scalars = data.x[:, 3:]
        scalars = torch.zeros(data.pos.shape[0], 1, device=data.pos.device)  # scalars cannot be None, so inputting zeros instead
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_oriented_plane(multivectors).squeeze()
        return output


def positional_encoding(data):
    data.x = torch.cat([data.norm, data.pos_enc], dim=1)
    return data


def main():
    transform = pyg.transforms.Compose((
        PointCloudPoolingScales(
            rel_sampling_ratios = (args.rel_sampling_ratio,),
            interp_simplex = args.interp_simplex
        ),
        positional_encoding
    ))
    
    device = torch.device("cpu")
    
    neural_network = LaBGATr(
        GeometricAlgebraInterface,
        d_model = args.d_model,
        num_blocks = args.num_blocks,
        num_attn_heads = args.num_attn_heads,
        pooling_mode = args.pooling_mode
    )
    neural_network.to(device)
    neural_network.load_state_dict(torch.load(args.model_weights, map_location=torch.device("cpu"), weights_only=True))
    neural_network.eval()

    datafile = torch.load(args.input_file, weights_only=False)
    input_points = datafile['input_points'].float()
    data = Data(
        pos = input_points,
        faces = datafile['input_faces'].int(),
        norm = datafile['input_normals'].float(),
        pos_enc = cartesian_to_positional_encoding(input_points)
    )
    data.to(device)
    data = transform(data)
    
    with torch.no_grad():
        prediction = neural_network(data)
        pred = data.pos + prediction
    
        save_dir = Path(args.input_file).parent
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred.numpy())
        o3d.io.write_point_cloud(save_dir / f"ct_scan_pred3.ply", pcd)


if __name__ == '__main__':
    main()

