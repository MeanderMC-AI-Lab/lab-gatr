import torch_geometric as pyg
import torch
import torch.nn.functional as F
import os
import math
from glob import glob
from tqdm import tqdm
from torch_geometric.data import Data
from pathlib import Path


def cartesian_to_positional_encoding(x: torch.Tensor, num_freq = 6,
                                     log_space = True) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    x = x.unsqueeze(-2)  # (..., 1, 3)
    if log_space:
        freq_bands = 2. ** torch.arange(num_freq, dtype=x.dtype, device=x.device)
    else:
        freq_bands = torch.linspace(1., 2. ** (num_freq - 1), num_freq,
                                    dtype=x.dtype, device=x.device)
    scaled = x * freq_bands.view(*([1] * (x.ndim - 2)), num_freq, 1) * math.pi
    sin = torch.sin(scaled)
    cos = torch.cos(scaled)
    per_freq = torch.cat([sin, cos], dim=-1)  # (..., L, 6)
    per_freq = per_freq.view(*orig_shape, -1)  # (..., 6 * L)
    return per_freq


def calc_angle(a, b):
        cross = torch.cross(a, b, dim=1)
        sin = cross.norm(dim=1).clamp_min(0.)
        cos = (a * b).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.atan2(sin, cos)


def angle_weighted_normals(vertices, faces):
    # Get vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Get edges
    e01 = v1 - v0
    e02 = v2 - v0
    e10 = v0 - v1
    e12 = v2 - v1
    e20 = v0 - v2
    e21 = v1 - v2

    # Calculate unit face normals
    cross_prod = torch.cross(e01, e02, dim=1)

    # Zero-out normals from degenerate faces
    valid = (cross_prod.norm(dim=1) > 1e-9).float().unsqueeze(1)
    face_normals = F.normalize(cross_prod, dim=1, eps=1e-6) * valid

    # Normalize edges
    e01n = F.normalize(e01, dim=1, eps=1e-6)
    e02n = F.normalize(e02, dim=1, eps=1e-6)
    e10n = F.normalize(e10, dim=1, eps=1e-6)
    e12n = F.normalize(e12, dim=1, eps=1e-6)
    e20n = F.normalize(e20, dim=1, eps=1e-6)
    e21n = F.normalize(e21, dim=1, eps=1e-6)

    # Calculate corner angles
    theta0 = calc_angle(e01n, e02n)
    theta1 = calc_angle(e10n, e12n)
    theta2 = calc_angle(e20n, e21n)

    # Accumualte weighted normals
    normals = torch.zeros_like(vertices)
    normals.index_add_(0, faces[:, 0], face_normals * theta0.unsqueeze(1))
    normals.index_add_(0, faces[:, 1], face_normals * theta1.unsqueeze(1))
    normals.index_add_(0, faces[:, 2], face_normals * theta2.unsqueeze(1))
    normals = F.normalize(normals, dim=1, eps=1e-6)

    # Orient normals towards the direction of the camera
    mask = normals[:, 2] > 0
    normals[mask] *= -1
    # invert = ((vertices * normals).sum(dim=1, keepdim=True) < 0).float()
    # normals = torch.where(flip, -normals, normals)
    # normals = F.normalize(normals, dim=1, eps=1e-6)
    return normals


class Dataset(pyg.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return sorted(glob(os.path.join(self.root, "raw", "*.pt")))

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(len(self.raw_file_names))]

    def download(self):
        return

    def process(self):
        for idx, path in enumerate(tqdm(self.raw_paths, desc="Reading & transforming", leave=False)):
            data = self.read_raw_datafile(path)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[idx]))

    @staticmethod
    def read_raw_datafile(path):
        datafile = torch.load(path, weights_only=False)

        # Load data
        input_points = datafile['input_points'].float() * 100  # Meters to centimeters
        input_faces = datafile['input_faces'].int()
        input_normals = datafile['input_normals'].float()
        displacements = datafile['displacements'].float() * 100
        target_points = datafile['target_points'].float() * 100
        # target_faces = datafile['target_faces'].int()
        target_normals = datafile['target_normals'].float()
        annotations_start = datafile['annotations_start'].float() * 100
        annotations_end = datafile['annotations_end'].float() * 100
        patient_features = datafile['patient_features'].float()

        # Transform data
        num_points = input_points.shape[0]
        positional_encoding = cartesian_to_positional_encoding(input_points)
        umbilicus_distances = input_points.norm(dim=1)
        umbilicus_vectors = -input_points / umbilicus_distances.clamp(min=1e-8).unsqueeze(1)
        patient_features = patient_features.unsqueeze(0).repeat(num_points, 1)
        # target_normals = angle_weighted_normals(target_points, target_faces)  # TODO: in creating "raw" data file, also store target_points such that they have same size as target_faces

        # Put into data file
        data = Data(
            pos = input_points,
            faces = input_faces,
            norm = input_normals,
            pos_enc = positional_encoding,
            y = displacements,
            pos_end = target_points,
            norm_end = target_normals,
            anns_start = annotations_start,
            anns_end = annotations_end,
            umb_dist = umbilicus_distances,
            umb_vec = umbilicus_vectors,
            pt_feat = patient_features
        )
        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
        return data
