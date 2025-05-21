import torch
from torch_scatter import scatter
from torch_cluster import knn
from prettytable import PrettyTable


def calc_closest_preds(labels_start, pcd_pos, pcd_preds, print_dist=False):
    idxs = knn(pcd_pos, labels_start, k=1)[1]
    if print_dist:
        mean_dist = torch.mean(torch.linalg.norm(labels_start - pcd_pos[idxs], dim=-1))
        print(f"Distance labels and start pcd: {mean_dist:.6f}")
    return pcd_preds[idxs]


def calc_chamfer_distance(pcd1, pcd2):
    dists = torch.cdist(pcd1, pcd2)
    dists1 = dists.min(dim=1)[0]
    dists2 = dists.min(dim=0)[0]
    return (dists1.mean() + dists2.mean()) / 2


class Evaluation():
    def __init__(self):
        self.values_dict = {
            'disps_gt': [],
            'disps_pred': [],
            'disps_idx': [],
            'anns_gt': [],
            'anns_pred': [],
            'anns_idx': [],
            'chamf_dist': []
        }
        self.results = {}

    def append_values(self, value_dict):
        for key, value in value_dict.items():
            if key == 'scatter_idx':
                self.values_dict['disps_idx'].append(value.expand(value_dict['disps_gt'].size(0)))
                self.values_dict['anns_idx'].append(value.expand(value_dict['anns_gt'].size(0)))
            elif key == 'chamf_dist':
                self.values_dict[key].append(value.unsqueeze(0))
            else:
                self.values_dict[key].append(value)

    def get_nmae(self, key_gt, key_pred, key_idx, normalize_max=False):
        if normalize_max:
            return self.get_mae(key_gt, key_pred, key_idx) / torch.max(torch.linalg.norm(self.values_dict[key_gt], dim=-1))
        l2_error = torch.linalg.norm(self.values_dict[key_gt] - self.values_dict[key_pred], dim=-1)
        magnitude = torch.linalg.norm(self.values_dict[key_gt], dim=-1).clamp(min=1e-8)
        nmae = scatter(
            l2_error / magnitude,
            self.values_dict[key_idx],
            dim=0,
            reduce='mean'
        )
        return nmae

    def get_mae(self, key_gt, key_pred, key_idx):
        mae = scatter(
            torch.linalg.norm(self.values_dict[key_gt] - self.values_dict[key_pred], dim=-1),
            self.values_dict[key_idx],
            dim=0,
            reduce='mean'
        )
        return mae

    def get_magnitude_error(self, key_gt, key_pred, key_idx):
        magn_gt = torch.linalg.norm(self.values_dict[key_gt], dim=-1).clamp(min=1e-8)
        magn_pred = torch.linalg.norm(self.values_dict[key_pred], dim=-1)
        magnitude_error = scatter(
            torch.abs(magn_gt - magn_pred) / magn_gt,
            self.values_dict[key_idx],
            dim=0,
            reduce='mean'
        )
        return magnitude_error

    def get_angle_error(self, key_gt, key_pred, key_idx):
        dot_products = (self.values_dict[key_pred] * self.values_dict[key_gt]).sum(dim=-1)
        denom = torch.linalg.norm(self.values_dict[key_pred], dim=-1) * torch.linalg.norm(self.values_dict[key_gt], dim=-1).clamp(min=1e-8)
        cos_angles = torch.clamp(dot_products / denom, -1., 1.)
        angle_error = scatter(
            torch.acos(cos_angles) * 180. / torch.pi,
            self.values_dict[key_idx],
            dim=0,
            reduce='mean'
        )
        return angle_error
            

    def get_approximation_error(self):
        approximation_error = torch.sqrt(scatter(
            torch.linalg.norm(self.values_dict['disps_gt'] - self.values_dict['disps_pred'],
                              dim=-1) ** 2,
            self.values_dict['disps_idx'],
            dim=0,
            reduce='sum'
        ) / scatter(
            torch.linalg.norm(self.values_dict['disps_gt'], dim=-1) ** 2,
            self.values_dict['disps_idx'],
            dim=0,
            reduce='sum'
        ))
        return approximation_error

    def get_mean_cosine_similarity(self):
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1).forward(
            self.values_dict['disps_gt'],
            self.values_dict['disps_pred']
        )
        mean_cosine_similarity = scatter(
            cosine_similarity,
            self.values_dict['disps_idx'],
            dim=0,
            reduce='mean'
        )
        return mean_cosine_similarity

    def calc_results(self):
        self.values_dict = {key: torch.cat(value, dim=0) for key, value in self.values_dict.items()}
        results = {
            'disps_mae': self.get_mae('disps_gt', 'disps_pred', 'disps_idx'),
            'disps_nmae': self.get_nmae('disps_gt', 'disps_pred', 'disps_idx'),
            'disps_magn': self.get_magnitude_error('disps_gt', 'disps_pred', 'disps_idx'),
            'disps_angle': self.get_angle_error('disps_gt', 'disps_pred', 'disps_idx'),
            'approximation_error': self.get_approximation_error(),
            'mean_cosine_similarity': self.get_mean_cosine_similarity(),
            'anns_mae': self.get_mae('anns_gt', 'anns_pred', 'anns_idx'),
            'anns_nmae': self.get_nmae('anns_gt', 'anns_pred', 'anns_idx'),
            'anns_angle': self.get_angle_error('anns_gt', 'anns_pred', 'anns_idx'),
            'anns_magn': self.get_magnitude_error('anns_gt', 'anns_pred', 'anns_idx')
        }
        return results

    def get_results(self):
        if not self.results:
            self.results = self.calc_results()
        return {
            'MAE_neuralode': torch.mean(self.results['disps_mae']).item(),
            'MAE_neuralode_std': torch.std(self.results['disps_mae']).item(),
            'NMAE_neuralode': torch.mean(self.results['disps_nmae']).item(),
            'NMAE_neuralode_std': torch.std(self.results['disps_nmae']).item(),
            'magn_neuralode': torch.mean(self.results['disps_magn']).item(),
            'magn_neuralode_std': torch.std(self.results['disps_magn']).item(),
            'angle_neuralode': torch.mean(self.results['disps_angle']).item(),
            'angle_neuralode_std': torch.std(self.results['disps_angle']).item(),
            'MAE_anns': torch.mean(self.results['anns_mae']).item(),
            'MAE_anns_std': torch.std(self.results['anns_mae']).item(),
            'NMAE_anns': torch.mean(self.results['anns_nmae']).item(),
            'NMAE_anns_std': torch.std(self.results['anns_nmae']).item(),
            'magn_anns': torch.mean(self.results['anns_magn']).item(),
            'magn_anns_std': torch.std(self.results['anns_magn']).item(),
            'angle_anns': torch.mean(self.results['anns_angle']).item(),
            'angle_anns_std': torch.std(self.results['anns_angle']).item(),
            'chamfer_distance': torch.mean(self.values_dict['chamf_dist']).item(),
            'chamfer_distance_std': torch.std(self.values_dict['chamf_dist']).item()
        }
            
    def make_table(self):
        if not self.results:
            self.results = self.calc_results()
            
        table = PrettyTable(["Metric", "Mean", "Standard Deviation"])

        table.add_row([
            "MAE (neuralODE)",
            "{0:.4f}".format(torch.mean(self.results['disps_mae']).item()),
            "{0:.4f}".format(torch.std(self.results['disps_mae']).item())
        ])

        table.add_row([
            "NMAE (neuralODE)",
            "{0:.2%}".format(torch.mean(self.results['disps_nmae']).item()),
            "{0:.2%}".format(torch.std(self.results['disps_nmae']).item())
        ])

        table.add_row([
            "Magnitude (neuralODE)",
            "{0:.2%}".format(torch.mean(self.results['disps_magn']).item()),
            "{0:.2%}".format(torch.std(self.results['disps_magn']).item()),
        ])

        table.add_row([
            "Angle (neuralODE)",
            "{0:.2f}".format(torch.mean(self.results['disps_angle']).item()),
            "{0:.2f}".format(torch.std(self.results['disps_angle']).item())
        ])

        table.add_row(["-"*10, "-"*5, "-"*5])

        table.add_row([
            "MAE (annotations)",
            "{0:.4f}".format(torch.mean(self.results['anns_mae']).item()),
            "{0:.4f}".format(torch.std(self.results['anns_mae']).item())
        ])

        table.add_row([
            "NMAE (annotations)",
            "{0:.2%}".format(torch.mean(self.results['anns_nmae']).item()),
            "{0:.2%}".format(torch.std(self.results['anns_nmae']).item())
        ])

        table.add_row([
            "Magnitude (annotations)",
            "{0:.2%}".format(torch.mean(self.results['anns_magn']).item()),
            "{0:.2%}".format(torch.std(self.results['anns_magn']).item()),
        ])

        table.add_row([
            "Angle (annotations)",
            "{0:.2f}".format(torch.mean(self.results['anns_angle']).item()),
            "{0:.2f}".format(torch.std(self.results['anns_angle']).item())
        ])

        table.add_row(["-"*10, "-"*5, "-"*5])

        table.add_row([
            "Chamfer distance",
            "{0:.4f}".format(torch.mean(self.values_dict['chamf_dist']).item()),
            "{0:.4f}".format(torch.std(self.values_dict['chamf_dist']).item())
        ])

        table.add_row([
            "Approximation Error",
            "{0:.2%}".format(torch.mean(self.results['approximation_error']).item()),
            "{0:.2%}".format(torch.std(self.results['approximation_error']).item())
        ])

        table.add_row([
            "Mean Cosine Similarity",
            "{:.3f}".format(torch.mean(self.results['mean_cosine_similarity']).item()),
            "{:.3f}".format(torch.std(self.results['mean_cosine_similarity']).item())
        ])

        return table


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        print(f"> Early stopping stats: {val_loss:.4f} (val loss) {self.best_loss:.4f} (best loss) {self.counter} (counter)")

