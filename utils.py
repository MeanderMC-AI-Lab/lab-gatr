import torch
from torch_scatter import scatter
from torch_cluster import knn
from prettytable import PrettyTable


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

    def append_values(self, value_dict):
        for key, value in value_dict.items():
            if key == 'scatter_idx':
                self.values_dict['disps_idx'].append(value.expand(value_dict['disps_gt'].size(0)))
                self.values_dict['anns_idx'].append(value.expand(value_dict['anns_gt'].size(0)))
            elif key == 'chamf_dist':
                self.values_dict[key].append(value.unsqueeze(0))
            else:
                self.values_dict[key].append(value)

    def lists_to_tensors(self):
        self.values_dict = {key: torch.cat(value, dim=0) for key, value in self.values_dict.items()}

    def get_nmae(self, key_gt, key_pred, key_idx):
        return self.get_mae(key_gt, key_pred, key_idx) / torch.max(torch.linalg.norm(
            self.values_dict[key_gt], dim=-1))

    def get_mae(self, key_gt, key_pred, key_idx):
        mae = scatter(
            torch.linalg.norm(self.values_dict[key_gt] - self.values_dict[key_pred], dim=-1),
            self.values_dict[key_idx],
            dim=0,
            reduce='mean'
        )
        return mae

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

    def make_table(self):
        self.lists_to_tensors()

        disps_mae = self.get_mae('disps_gt', 'disps_pred', 'disps_idx')
        disps_nmae = self.get_nmae('disps_gt', 'disps_pred', 'disps_idx')
        approximation_error = self.get_approximation_error()
        mean_cosine_similarity = self.get_mean_cosine_similarity()
        anns_mae = self.get_mae('anns_gt', 'anns_pred', 'anns_idx')
        anns_nmae = self.get_nmae('anns_gt', 'anns_pred', 'anns_idx')

        table = PrettyTable(["Metric", "Mean", "Standard Deviation"])

        table.add_row([
            "MAE",
            "{0:.4f}".format(torch.mean(disps_mae).item()),
            "{0:.4f}".format(torch.std(disps_mae).item())
        ])

        table.add_row([
            "NMAE",
            "{0:.2%}".format(torch.mean(disps_nmae).item()),
            "{0:.2%}".format(torch.std(disps_nmae).item())
        ])
        
        table.add_row([
            "Approximation Error",
            "{0:.2%}".format(torch.mean(approximation_error).item()),
            "{0:.2%}".format(torch.std(approximation_error).item())
        ])

        table.add_row([
            "Mean Cosine Similarity",
            "{:.3f}".format(torch.mean(mean_cosine_similarity).item()),
            "{:.3f}".format(torch.std(mean_cosine_similarity).item())
        ])

        table.add_row([
            "MAE (annotations)",
            "{0:.4f}".format(torch.mean(anns_mae).item()),
            "{0:.4f}".format(torch.std(anns_mae).item())
        ])

        table.add_row([
            "NMAE (annotations)",
            "{0:.4%}".format(torch.mean(anns_nmae).item()),
            "{0:.4%}".format(torch.std(anns_nmae).item())
        ])

        table.add_row([
            "Chamfer distance",
            "{0:.4f}".format(torch.mean(self.values_dict['chamf_dist']).item()),
            "{0:.4f}".format(torch.std(self.values_dict['chamf_dist']).item())
        ])

        return table


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

