import torch_geometric as pyg
import torch
import os
from glob import glob
from tqdm import tqdm
from torch_geometric.data import Data
from pathlib import Path


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

        # Calculate vectors from all points to umbilicus
        input_points = datafile['input_points'].float()
        umb_dists = input_points.norm(dim=1)
        umb_vecs = -input_points / umb_dists.clamp(min=1e-8).unsqueeze(1)
        long_pos = input_points[:, 1]
        num_points = input_points.shape[0]
        pt_feat = datafile['patient_features'].float().unsqueeze(0).repeat(num_points, 1)
        
        data = Data(
            y = datafile['displacements'].float(),               # NeuralODE displacement samples
            pos = input_points,                                  # Measured pointcloud before insufflation
            norm = datafile['input_normals'].float(),            # Vertex normals calculated from mesh
            pt_feat = pt_feat,                                   # Patient features, such as length
            anns_start = datafile['annotations_start'].float(),  # Labelled locations on not-insufflated abdomen
            anns_end = datafile['annotations_end'].float(),      # Labelled locations on insufflated abdomen
            pos_end = datafile['target_points'].float(),         # Measured pointcloud after insufflation
            umb_dist = umb_dists,                                # Distance from the umbilicus
            umb_vec = umb_vecs,                                  # Unit vector in the direction of the umbilicus
            long_pos = long_pos                                  # Scalar as position on longitudinal axis
        )
        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
        return data
