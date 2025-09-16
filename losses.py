import torch
import torch.nn.functional as F
import ot
import torch_geometric as pyg
from chamferdist import ChamferDistance
from torch_scatter import scatter_add
from torch_cluster import knn

import pdb


class L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + 1e-8)


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.chamfer = ChamferDistance()

    def forward(self, yhat, y):
        loss = self.chamfer(yhat, y, bidirectional=True, point_reduction='mean')
        return torch.sqrt(.5 * loss)


class SlicedWasserStein(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ws_func = ot.sliced.sliced_wasserstein_distance
        # self.ws_func = ot.sliced.max_sliced_wasserstein_distance
        # self.ws_func = ot.sliced.sliced_wasserstein_sphere

    def forward(self, yhat, y):
        return self.ws_func(yhat, y, n_projections=50, p=2)


class LaplacianLoss(torch.nn.Module):

    def __init__(self, k=16):
        super().__init__()
        self.k = k

    @torch.no_grad()
    def build_graph(self, x0):
        edge_index = pyg.nn.knn_graph(x0, k=self.k, batch=None, loop=False)
        return pyg.utils.get_laplacian(edge_index, normalization=None, num_nodes=x0.size(0))
        
    def forward(self, x0, x):
        (row, col), edge_weight = self.build_graph(x0)
        edge_weight = edge_weight.view(-1, 1)
        y = scatter_add(edge_weight * x[col], row, dim=0, dim_size=x.size(0))
        return (y**2).sum(dim=1).mean()


class NormalLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def _calc_angle(self, a, b):
        cross = torch.cross(a, b, dim=1)
        sin = cross.norm(dim=1).clamp_min(0.)
        cos = (a * b).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.atan2(sin, cos)

    def _calc_angle_weighted_normals(self, vertices, faces):
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
        theta0 = self._calc_angle(e01n, e02n)
        theta1 = self._calc_angle(e10n, e12n)
        theta2 = self._calc_angle(e20n, e21n)

        # Accumualte weighted normals
        normals = torch.zeros_like(vertices)
        normals.index_add_(0, faces[:, 0], face_normals * theta0.unsqueeze(1))
        normals.index_add_(0, faces[:, 1], face_normals * theta1.unsqueeze(1))
        normals.index_add_(0, faces[:, 2], face_normals * theta2.unsqueeze(1))
        normals = F.normalize(normals, dim=1, eps=1e-6)

        # Orient normals towards the direction of the camera
        mask = normals[:, 2] > 0
        normals[mask] *= -1
        return normals
        
    def _calc_normals(self, vertices, faces):
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

    @torch.no_grad()
    def _find_reference_normals(self, vert0, vert1, normals):
        _, col = knn(x=vert1, y=vert0, k=1)
        return normals[col]

    def forward(self, x, faces, vertices, normals):
        pred_norm = self._calc_normals(x, faces)
        # pred_norm = self._calc_angle_weighted_normals(x, faces)
        ref_norm = self._find_reference_normals(x, vertices, normals)
        cos = F.cosine_similarity(pred_norm, ref_norm, dim=1)
        return (1. - cos).mean()

