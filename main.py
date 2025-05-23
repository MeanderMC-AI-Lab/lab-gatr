from argparse import ArgumentParser

from lab_gatr.transforms import PointCloudPoolingScales
import torch_geometric as pyg
from datasets import Dataset
import wandb_impostor as wandb
from lab_gatr import LaBGATr
# from lab_gatr.models import LaBVaTr  # geometric algebra ablated LaB-GATr
import torch
from torch_cluster import knn
from gatr.interface import embed_point, embed_oriented_plane, extract_oriented_plane, extract_translation
import os
from tqdm import tqdm
import statistics
from torch.nn.parallel import DistributedDataParallel
from functools import partial
from utils import Evaluation, calc_closest_preds, calc_chamfer_distance
import meshio
import sys
from time import asctime
from visualisation import save_pred_and_gt_pointclouds
from chamferdist import ChamferDistance


parser = ArgumentParser()
parser.add_argument('--data_root', type=str, default='/data/Predict-Pneumoperitoneum_LaB-GATr/dataset')
parser.add_argument('--pooling_mode', type=str, default='cross_attention')  # message_passing, cross_attention
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=0)
parser.add_argument('--run_id', type=str, default=None)
args = parser.parse_args()


wandb_config = {
    'batch_size': 1,
    'learning_rate': 3e-4,  # best learning rate for Adam, hands down
    'num_epochs': args.num_epochs,
    'lr_decay_gamma': 0.9989,
    'interp_simplex': 'triangle'  # triangle or tetrahedron
}


def main(rank, num_gpus):
    ddp_setup(rank, num_gpus, project_name="lab_gatr", wandb_config=wandb_config, run_id=args.run_id)

    dataset = Dataset(args.data_root, pre_transform=pyg.transforms.Compose((
        PointCloudPoolingScales(rel_sampling_ratios=(0.01,), interp_simplex=wandb_config['interp_simplex']),
        # positional_encoding
    )))

    # Split: 52 train, 6 validation, 7 test
    training_data_loader = pyg.loader.DataLoader(
        dataset[get_dataset_slices_for_gpus(num_gpus, num_samples=52)[rank]],
        batch_size=wandb.config['batch_size'],
        shuffle=True
    )
    validation_data_loader = pyg.loader.DataLoader(
        dataset[get_dataset_slices_for_gpus(num_gpus, num_samples=6, first_sample_idx=52)[rank]],
        batch_size=wandb.config['batch_size'],
        shuffle=True
    )
    test_dataset_slice = slice(58, 65)
    visualisation_dataset_range = range(58, 65)

    neural_network = LaBGATr(GeometricAlgebraInterface, d_model=8, num_blocks=10, num_attn_heads=4, pooling_mode=args.pooling_mode)
    # neural_network = LaBVaTr(num_input_channels=12, num_output_channels=3, d_model=128, num_blocks=12, num_attn_heads=8)

    training_device = torch.device(f'cuda:{rank}')
    neural_network.to(training_device)

    load_neural_network_weights(neural_network, working_directory=f"exp{'-' if args.run_id else ''}{args.run_id or ''}")

    # Distributed data parallel (multi-GPU training)
    neural_network = ddp_module(neural_network, rank)

    wandb.watch(neural_network)
    training_loop(
        rank,
        neural_network,
        training_device,
        training_data_loader,
        validation_data_loader,
        working_directory=f"exp{'-' if args.run_id else ''}{args.run_id or ''}"
    )

    ddp_rank_zero(
        test_loop,
        neural_network=neural_network,
        training_device=training_device,
        dataset=dataset,
        test_dataset_slice=test_dataset_slice,
        visualisation_dataset_range=visualisation_dataset_range,
        working_directory=f"exp{'-' if args.run_id else ''}{args.run_id or ''}"
    )

    ddp_cleanup()


# @torch.no_grad()
# def positional_encoding(data):

#     vectors_to = {key: data.pos[value.long()] - data.pos for key, value in compute_nearest_boundary_vertex(data).items()}
#     distances_to = {key: torch.linalg.norm(value, dim=-1, keepdim=True) for key, value in vectors_to.items()}

#     data.x = torch.cat((
#         vectors_to['inlet'] / torch.clamp(distances_to['inlet'], min=1e-16),
#         vectors_to['lumen_wall'] / torch.clamp(distances_to['lumen_wall'], min=1e-16),
#         vectors_to['outlets'] / torch.clamp(distances_to['outlets'], min=1e-16),
#         distances_to['inlet'],
#         distances_to['lumen_wall'],
#         distances_to['outlets']
#     ), dim=1)

#     return data


# def compute_nearest_boundary_vertex(data):
#     index_dict = {}

#     for key in ('inlet', 'lumen_wall', 'outlets'):
#         index_dict[key] = data[f'{key}_index'][knn(data.pos[data[f'{key}_index'].long()], data.pos, k=1)[1].long()]

#     return index_dict


# class GeometricAlgebraInterface:
#     num_input_channels = 1 + 3  # vertex positions plus positional encoding vectors
#     num_output_channels = 1

#     num_input_scalars = 3  # positional encoding sclars
#     num_output_scalars = None

#     @staticmethod
#     @torch.no_grad()
#     def embed(data):

#         multivectors = torch.cat((
#             embed_point(data.pos).view(-1, 1, 16),
#             *(embed_oriented_plane(data.x[:, slice(i * 3, i * 3 + 3)], data.pos).view(-1, 1, 16) for i in range(3))
#         ), dim=1)
#         scalars = data.x[:, 9:]

#         return multivectors, scalars

#     @staticmethod
#     def dislodge(multivectors, scalars):
#         return extract_oriented_plane(multivectors).squeeze()



class GeometricAlgebraInterface:
    num_input_channels = 1  # vertex normals + pos
    num_output_channels = 1
    num_input_scalars = 1
    num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data):
        multivectors = embed_oriented_plane(normal=data.norm, position=data.pos).view(-1, 1, 16)
        scalars = torch.zeros(data.pos.shape[0], 1, device=data.pos.device)

        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_oriented_plane(multivectors).squeeze()
        # output = extract_translation(multivectors).squeeze()

        return output


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

    loss_function = torch.nn.L1Loss()

    optimiser = torch.optim.Adam(neural_network.parameters(), lr=wandb.config['learning_rate'])
    load_optimiser_state(rank, optimiser, working_directory)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=wandb.config['lr_decay_gamma'])

    for epoch in tqdm(range(wandb.config['num_epochs']), desc="Epochs", position=0, leave=True):

        loss_values = {'training': [], 'validation': []}

        # Objective convergence
        neural_network.train()

        for batch in tqdm(training_data_loader, desc="Training split", position=1, leave=False):
            optimiser.zero_grad()

            batch = batch.to(training_device)
            prediction = neural_network(batch)

            loss_value = loss_function(prediction, batch.y)
            loss_values['training'].append(loss_value.item())

            loss_value.backward()  # "autograd" hook fires and triggers gradient synchronisation across processes
            torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm=1.0, error_if_nonfinite=True)

            optimiser.step()

            del batch, prediction

        scheduler.step()

        ddp_rank_zero(save_neural_network_weights, neural_network, working_directory)
        torch.save(optimiser.state_dict(), os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt"))

        # Learning task
        neural_network.eval()

        with torch.no_grad():
            for batch in tqdm(validation_data_loader, desc="Validation split", position=1, leave=False):

                batch = batch.to(training_device)
                prediction = neural_network(batch)

                loss_value = loss_function(prediction, batch.y)
                loss_values['validation'].append(loss_value.item())

                del batch, prediction

        wandb.log({key: statistics.mean(value) for key, value in loss_values.items()})


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
    chamf_dist = ChamferDistance()
    neural_network.eval()

    with torch.no_grad():

        # Quantitative
        for i, data in enumerate(tqdm(dataset[test_dataset_slice], desc="Test split", position=0, leave=False)):
            data = data.to(training_device)
            prediction = neural_network(data)
            label_prediction = calc_closest_preds(data.anns_start, data.pos, prediction)
            # chamf_dist = calc_chamfer_distance((data.pos.cpu() + prediction.cpu()),
            #                                    data.pos_end.cpu())
            cd = chamf_dist((data.pos + prediction).unsqueeze(0),
                             data.pos_end.unsqueeze(0), bidirectional=True,
                             point_reduction='mean')
            cd = torch.sqrt(.5 * cd)
            
            evaluation.append_values({
                'disps_gt': data.y.cpu(),
                'disps_pred': prediction.cpu(),
                'anns_gt': (data.anns_end - data.anns_start).cpu(),
                'anns_pred': label_prediction.cpu(),
                'scatter_idx': torch.tensor(i),
                'chamf_dist': cd
            })

            del data

        print(f"{evaluation.make_table()}")
        
        # Qualitative (visual)
        # neural_network.cpu()  # un-comment to avoid memory issues

        for idx in tqdm(visualisation_dataset_range, desc="Visualisation split", position=0, leave=False):
            data = dataset.__getitem__(idx).to(training_device)  # avoid "Floating point exception"
            data.Y = neural_network(data)

            if working_directory and not os.path.exists(working_directory):
                os.makedirs(working_directory)

            data.cpu()
            # meshio.Mesh(data.pos, [('tetra', data.tets.T)], point_data={
            #     'reference': data.y,
            #     'prediction': data.Y,
            #     'clusters': data.scale0_pool_target
            # }).write(os.path.join(working_directory, f"visuals_idx_{idx:04d}.vtu"))
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
