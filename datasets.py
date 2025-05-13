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
        data = Data(
            y=datafile['displacements'].float(),
            pos=datafile['input_points'].float(),
            norm=datafile['input_normals'].float(),
            feat=datafile['patient_features'].float()
        )
        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
        return data
