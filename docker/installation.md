# Install environment for LaB-GATr

- Image: ```mmc-pytorch:2024.03c```
- Install PyTorch, xFormers, PyG: ```pip install torch xformers torch_geometric torchvision==0.22.0 torchaudio==2.7.0```
- Install other dependencies: ```pip install torch_scatter torch_cluster --find-links https://data.pyg.org/whl/torch-2.6.0+cu126.html```
- Install GATr: ```pip install git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git```
- Install LaB-GATr: ```pip install .``` inside main folder
- Install: ```pip install prettytable meshio chamferdist```

# Install dependencies for loading meshes

- Open3D: ```pip install open3d```