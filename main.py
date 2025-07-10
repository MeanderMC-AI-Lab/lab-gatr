from argparse import ArgumentParser

from lab_gatr.transforms import PointCloudPoolingScales
import torch_geometric as pyg
from torch.utils.data import Subset
from datasets import Dataset
import wandb_impostor as wandb
# import wandb
from lab_gatr import LaBGATr
import torch
from torch_cluster import knn
from gatr.interface import embed_point, embed_oriented_plane, extract_oriented_plane, extract_translation
import os
from tqdm import tqdm
import statistics
from torch.nn.parallel import DistributedDataParallel
from functools import partial
from utils import Evaluation, calc_closest_preds, calc_chamfer_distance, EarlyStopping
import meshio
import sys
import json
from time import asctime
from visualisation import save_pred_and_gt_pointclouds
from chamferdist import ChamferDistance
from sklearn.model_selection import KFold
from lab_gatr.nn.mlp.vanilla import MLP
from torch_dvf.models import PointNet, SEPointNet
from torch_dvf.transforms import RadiusPointCloudHierarchy
from e3nn import o3


def calculate_inputs():
    multivectors = 0
    scalars = 0
    if args.feat_norm:
        multivectors += 1
    if args.feat_umbilicus:
        multivectors += 1
        scalars += 1
    if args.feat_longitudinal:
        scalars += 1
    return multivectors, scalars


parser = ArgumentParser()
# Run settings
parser.add_argument('--data_root', type=str, default='/data/Predict-Pneumoperitoneum_LaB-GATr/dataset')
parser.add_argument('--run_id', type=str, default="sweep")
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=3e-4)  # best learning rate for Adam, hands down
parser.add_argument('--lr_decay_gamma', type=float, default=0.9989)
parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'chamfer'], default='l2')
# Model settings
parser.add_argument('--rel_sampling_ratio', type=float, default=0.01)
parser.add_argument('--model', type=str, choices=['labgatr', 'pointnet', 'sepointnet', 'mlp'], default='labgatr')
parser.add_argument('--interp_simplex', type=str, choices=['triangle', 'tetrahedron'], default='tetrahedron')
parser.add_argument('--pooling_mode', type=str, choices=['message_passing', 'cross_attention'], default='cross_attention')
parser.add_argument('--d_model', type=int, default=8)
parser.add_argument('--num_blocks', type=int, default=12)
parser.add_argument('--num_attn_heads', type=int, default=8)
parser.add_argument('--num_latent_channels', type=int, default=32)
# Features settings
parser.add_argument('--feat_norm', action='store_true')
parser.add_argument('--feat_umbilicus', action='store_true')
parser.add_argument('--feat_longitudinal', action='store_true')
args = parser.parse_args()
wandb_config = vars(args)
# Calculate number of point features
args.multivectors, args.scalars = calculate_inputs()
args.num_input_channels = args.multivectors * 3 + args.scalars


@torch.no_grad()
def positional_encoding(data):
    features = []
    if args.feat_norm:
        features.append(data.norm)
    if args.feat_umbilicus:
        features.append(data.umb_vec)
        features.append(data.umb_dist.unsqueeze(1))
    if args.feat_longitudinal:
        features.append(data.long_pos.unsqueeze(1))
    data.x = torch.cat(features, dim=1)
    return data


class GeometricAlgebraInterface:
    num_input_channels = args.multivectors + 1  # Add one for data.pos
    num_input_scalars = args.scalars if args.scalars > 0 else 1
    num_output_channels = 1
    num_output_scalars = None

    @staticmethod
    @torch.no_grad()
    def embed(data):
        multivectors = torch.cat((
            embed_point(data.pos).view(-1, 1, 16),
            *(embed_oriented_plane(data.x[:, slice(i * 3, i * 3 + 3)], data.pos).view(-1, 1, 16) for i in range(args.multivectors))
        ), dim=1)
        if args.scalars == 0:
            scalars = torch.zeros(data.pos.shape[0], 1, device=data.pos.device)  # scalars cannot be None, so inputting zeros instead
        else:
            scalars = data.x[:, args.multivectors*3:]
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_oriented_plane(multivectors).squeeze()
        return output


def make_splits(dataset, num_folds, shuffle, load_from_json=False):
    # Single fold
    if num_folds==1:
        train_idx = [31, 8, 16, 40, 62, 11, 35, 5, 17, 27, 52, 18, 21, 60, 34, 14, 48, 42, 12, 61,
                     36, 63, 0, 41, 1, 39, 55, 2, 9, 23, 47, 20, 13, 3, 53, 43, 15, 4, 26, 24,
                     59, 10, 25, 33, 45, 30, 54, 28, 51, 38, 50, 44]
        val_idx = [58, 32, 49, 6, 64, 57]
        test_idx = [46, 22, 29, 37, 56, 19, 7]
        return [(train_idx, val_idx, test_idx)]

    # Load predefined folds from json file
    if load_from_json:
        assert num_folds in [5, 10]
        with open("folds.json", "r") as f:
            fold_idxs = json.load(f)[f"{num_folds}-fold"]
        with open("map_patient_idx.json", "r") as f:
            map_idxs = json.load(f)
        
        folds = []
        for fold_id in range(num_folds):
            train_idx, val_idx, test_idx = [], [], []
            for n in range(num_folds):
                patient_idxs = fold_idxs[str(n)]
                sample_idxs = [map_idxs[str(i)] for i in patient_idxs]
                if n==fold_id:
                    test_idx.extend(sample_idxs)
                elif n==fold_id+1 or (fold_id==num_folds-1 and n==0):
                    val_idx.extend(sample_idxs)
                else:
                    train_idx.extend(sample_idxs)
            folds.append((train_idx, val_idx, test_idx))
        return folds

    # Create folds randomly
    else:
        kfold = KFold(n_splits=num_folds, shuffle=shuffle)  # random_state=42
        folds = []
        for (train_idx, test_idx) in kfold.split(dataset):
            num_val_samples = len(test_idx)            
            val_idx = train_idx[-num_val_samples:]
            train_idx = train_idx[:-num_val_samples]
            folds.append((train_idx, val_idx, test_idx))
        return folds


def select_model(model_type):
    if model_type == 'labgatr':
        return LaBGATr(
            GeometricAlgebraInterface,
            d_model = args.d_model,
            num_blocks = args.num_blocks,
            num_attn_heads = args.num_attn_heads,
            pooling_mode = args.pooling_mode
        )
    if model_type == 'pointnet':
        return PointNet(
            num_input_channels = args.num_input_channels,
            num_output_channels = 3,
            num_hierarchies = 1,
            num_latent_channels = args.num_latent_channels
        )
    if model_type == 'sepointnet':
        input_irreps = ""
        if args.multivectors > 0:
            input_irreps = f"{args.multivectors}x1o"
            if args.scalars > 0:
                input_irreps += "+"
        if args.scalars > 0:
            input_irreps += f"{args.scalars}x0e"
        return SEPointNet(
            o3.Irreps(input_irreps),
            o3.Irreps("1x1o"),
            num_hierarchies = 1,
            num_latent_channels = args.num_latent_channels
        )
    else:
        channels_in = args.num_input_channels + 3
        return MLP(
            (channels_in, 512, 512, 512, 512, 512, 512, 3),
            plain_last = True,
            use_norm_in_first = True,
            dropout_probability = False,
            use_running_stats_in_norm = True
        )

def select_transform(model_type):
    if model_type in ['labgatr']:
        return pyg.transforms.Compose((
            PointCloudPoolingScales(
                rel_sampling_ratios = (args.rel_sampling_ratio,),
                interp_simplex = args.interp_simplex
            ),
            positional_encoding
        ))
    if model_type in ['pointnet', 'sepointnet']:
        return pyg.transforms.Compose((
            RadiusPointCloudHierarchy(
                rel_sampling_ratios = (args.rel_sampling_ratio,),
                cluster_radii = None,
                interp_simplex = args.interp_simplex
            ),
            positional_encoding
        ))
    else:
        return None


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


def main(rank, num_gpus):
    assert num_gpus == 1

    # Build dataset with folds for cross-validation
    transform = select_transform(args.model)
    dataset = Dataset(args.data_root, pre_transform=transform)
    data_splits = make_splits(dataset, wandb_config['num_folds'], shuffle=True, load_from_json=True)
    
    for fold, (train_idx, val_idx, test_idx) in enumerate(data_splits):

        # Setup fold
        wandb_config['fold'] = fold
        print(f"Fold {fold}: {len(train_idx)} (train) {len(val_idx)} (val) {len(test_idx)} (test)")
        working_dir = os.path.join(f"exp{'-' if args.run_id else ''}{args.run_id or ''}", f"fold{fold}")
        ddp_setup(rank, num_gpus, project_name="lab_gatr", wandb_config=wandb_config,
                  run_id=args.run_id+f"_fold{fold}")
        training_device = torch.device(f'cuda:{rank}')

        # Make train and validation dataloaders
        training_data_loader = pyg.loader.DataLoader(
            Subset(dataset, train_idx),
            batch_size=wandb.config['batch_size'],
            shuffle=True
        )
        validation_data_loader = pyg.loader.DataLoader(
            Subset(dataset, val_idx),
            batch_size=wandb.config['batch_size'],
            shuffle=True
        )
        test_dataset_slice = test_idx
        visualisation_dataset_range = test_idx

        # Build neural network and load weights
        neural_network = select_model(args.model)
        neural_network.to(training_device)
        load_neural_network_weights(neural_network, working_directory=working_dir)
    
        # Distributed data parallel (multi-GPU training)
        neural_network = ddp_module(neural_network, rank)

        # Training and validation
        wandb.watch(neural_network)
        training_loop(
            rank,
            neural_network,
            training_device,
            training_data_loader,
            validation_data_loader,
            working_directory=working_dir
        )

        # Testing
        ddp_rank_zero(
            test_loop,
            neural_network=neural_network,
            training_device=training_device,
            dataset=dataset,
            test_dataset_slice=test_dataset_slice,
            visualisation_dataset_range=visualisation_dataset_range,
            working_directory=working_dir
        )
    
        ddp_cleanup()


def get_dataset_slices_for_gpus(num_gpus, num_samples, first_sample_idx=0):
    per_gpu = int(num_samples / num_gpus)

    first_and_last_idx_per_gpu = tuple(zip(
        range(first_sample_idx, first_sample_idx + num_samples - per_gpu + 1, per_gpu),
        range(first_sample_idx + per_gpu, first_sample_idx + num_samples + 1, per_gpu)
    ))

    return [slice(*idcs) for idcs in first_and_last_idx_per_gpu]


def load_neural_network_weights(neural_network, working_directory=""):
    if os.path.exists(os.path.join(working_directory, "neural_network_weights.pt")):

        neural_network.load_state_dict(torch.load(os.path.join(working_directory, "neural_network_weights.pt")))
        print("Resuming from pre-trained neural-network weights.")


def training_loop(rank, neural_network, training_device, training_data_loader, validation_data_loader, working_directory):

    loss_options = {'l1': torch.nn.L1Loss(), 'l2': L2Loss(), 'chamfer': ChamferLoss()}
    loss_function = loss_options[args.loss]

    optimiser = torch.optim.Adam(neural_network.parameters(), lr=wandb.config['learning_rate'])
    load_optimiser_state(rank, optimiser, working_directory)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=wandb.config['lr_decay_gamma'])
    early_stopping = EarlyStopping(patience=3, min_delta=0.0001)

    for epoch in tqdm(range(wandb.config['num_epochs']), desc="Epochs", position=0, leave=True):

        loss_values = {'training': [], 'validation': []}

        # Objective convergence
        neural_network.train()

        for batch in tqdm(training_data_loader, desc="Training split", position=1, leave=False):
            optimiser.zero_grad()

            batch = batch.to(training_device)
            if args.model == 'mlp':
                prediction = neural_network(torch.cat([batch.pos, batch.x], dim=1))
            else:
                prediction = neural_network(batch)
                
            if args.loss in ['l1', 'l2']:
                loss_value = loss_function(prediction, batch.y)
            else:
                loss_value = loss_function((batch.pos + prediction).unsqueeze(0),
                                           batch.pos_end.unsqueeze(0))
            
            loss_values['training'].append(loss_value.item())
            loss_value.backward()  # "autograd" hook fires and triggers gradient synchronisation across processes
            torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimiser.step()

            del batch, prediction

        scheduler.step()

        if args.run_id != "sweep":
            ddp_rank_zero(save_neural_network_weights, neural_network, working_directory)
            torch.save(optimiser.state_dict(), os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt"))

        # Learning task
        neural_network.eval()

        with torch.no_grad():
            for batch in tqdm(validation_data_loader, desc="Validation split", position=1, leave=False):

                batch = batch.to(training_device)
                if args.model == 'mlp':
                    prediction = neural_network(torch.cat([batch.pos, batch.x], dim=1))
                else:
                    prediction = neural_network(batch)
                target = batch.y if args.loss in ['l1', 'l2'] else batch.pos_end

                if args.loss in ['l1', 'l2']:
                    loss_value = loss_function(prediction, batch.y)
                else:
                    loss_value = loss_function((batch.pos + prediction).unsqueeze(0),
                                               batch.pos_end.unsqueeze(0))
                loss_values['validation'].append(loss_value.item())

                del batch, prediction

        wandb.log({key: statistics.mean(value) for key, value in loss_values.items()} | {'epoch': epoch})
        early_stopping(statistics.mean(loss_values['validation']))
        if early_stopping.early_stop:
            print(f"Early stopping training with: {statistics.mean(loss_values['validation']):.4f} (val loss)")
            break


def load_optimiser_state(rank, optimiser, working_directory=""):
    if os.path.exists(os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt")):

        optimiser.load_state_dict(torch.load(os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt")))
        print("Resuming from previous optimiser state.")


def save_neural_network_weights(neural_network, working_directory="", file_name=None):

    if isinstance(neural_network, DistributedDataParallel):
        neural_network_weights = neural_network.module.state_dict()
    else:
        neural_network_weights = neural_network.state_dict()

    if working_directory and not os.path.exists(working_directory):
        os.makedirs(working_directory)

    torch.save(neural_network_weights, os.path.join(working_directory, file_name or "neural_network_weights.pt"))


def test_loop(neural_network, training_device, dataset, test_dataset_slice, visualisation_dataset_range, working_directory):
    evaluation = Evaluation()
    chamfer_loss = ChamferLoss()
    neural_network.eval()

    with torch.no_grad():

        # Quantitative
        for i, data in enumerate(tqdm(dataset[test_dataset_slice], desc="Test split", position=0, leave=False)):
            data = data.to(training_device)
            if args.model == 'mlp':
                prediction = neural_network(torch.cat([data.pos, data.x], dim=1))
            else:
                prediction = neural_network(batch)
            label_prediction = calc_closest_preds(data.anns_start, data.pos, prediction)
            chamfer = chamfer_loss((data.pos + prediction).unsqueeze(0),
                                   data.pos_end.unsqueeze(0))
            
            evaluation.append_values({
                'disps_gt': data.y.cpu(),
                'disps_pred': prediction.cpu(),
                'anns_gt': (data.anns_end - data.anns_start).cpu(),
                'anns_pred': label_prediction.cpu(),
                'scatter_idx': torch.tensor(i),
                'chamf_dist': chamfer
            })

            del data

        wandb.log(evaluation.get_results())
        print(f"{evaluation.make_table()}")
        
        # Qualitative (visual)
        # neural_network.cpu()  # un-comment to avoid memory issues
        if args.run_id != "sweep":
            for idx in tqdm(visualisation_dataset_range, desc="Visualisation split", position=0, leave=False):
                data = dataset.__getitem__(idx).to(training_device)  # avoid "Floating point exception"
                if args.model == 'mlp':
                    data.Y = neural_network(torch.cat([data.pos, data.x], dim=1))
                else:
                    data.Y = neural_network(data)
    
                if working_directory and not os.path.exists(working_directory):
                    os.makedirs(working_directory)
    
                data.cpu()
                save_pred_and_gt_pointclouds(working_directory, data.pos.numpy(),
                    data.y.numpy(), data.Y.numpy(), idx)


def ddp_setup(rank, num_gpus, project_name, wandb_config, run_id=None):

    if num_gpus > 1:

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        sys.stderr = open(f"{project_name}{'_' if run_id else ''}{run_id or ''}_{rank}.out", 'w')  # used by "tqdm"

        torch.distributed.init_process_group('nccl', rank=rank, world_size=num_gpus)
        wandb.init(project=project_name, config=wandb_config, group=f"{run_id or 'DDP'} ({asctime()})")

    else:
        wandb.init(project=project_name, config=wandb_config, name=run_id)


def ddp_module(torch_module, rank):
    return DistributedDataParallel(torch_module, device_ids=[rank]) if torch.distributed.is_initialized() else torch_module


def ddp_rank_zero(fun, *args, **kwargs):

    if torch.distributed.is_initialized():
        fun(*args, **kwargs) if torch.distributed.get_rank() == 0 else None

        torch.distributed.barrier()  # synchronises all processes

    else:
        fun(*args, **kwargs)


def ddp_cleanup():
    wandb.finish()
    torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None
    sys.stderr.close() if torch.distributed.is_initialized() else None  # last executed statement


def ddp(fun, num_gpus):
    torch.multiprocessing.spawn(fun, args=(num_gpus,), nprocs=num_gpus, join=True) if num_gpus > 1 else fun(rank=0, num_gpus=num_gpus)


if __name__ == '__main__':
    ddp(main, args.num_gpus)
