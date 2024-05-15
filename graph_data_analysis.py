import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net1, TransferNet
import numpy as np
from utils import load_data, coarsening, create_distribution_tensor
import os
from tqdm import tqdm
import time
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
parser.add_argument('--runs', type=int, default=50)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--epochs1', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=300)
parser.add_argument('--num_layers1', type=int, default=2)
parser.add_argument('--num_layers2', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--extra_node', type=bool, default=True)
parser.add_argument('--cluster_node', type=bool, default=False)
parser.add_argument('--super_graph', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
# parser.add_argument('--output_dir', type=str, required=False)
args = parser.parse_args()

if args.super_graph:
    args.cluster_node = False
    args.extra_node = False
elif args.cluster_node:
    args.extra_node = False
    args.super_graph = False
elif args.extra_node:
    args.cluster_node = False
    args.super_graph = False

print(args)
args.num_features, args.num_classes, candidate, C_list, Gc_list, subgraph_list = coarsening(args, 1-args.coarsening_ratio, args.coarsening_method)

num_extra_nodes = []
num_orig_nodes = []
num_subgraph_nodes = []
for subgraph in subgraph_list:
    num_extra_nodes.append(len(subgraph.actual_ext))
    num_orig_nodes.append(subgraph.x.shape[0] - len(subgraph.actual_ext))
    num_subgraph_nodes.append(subgraph.x.shape[0])
                          
with open("results.txt", 'a') as f:
    f.write(f"{args.dataset}, {args.coarsening_ratio}, {len(num_extra_nodes)}, {np.sum(num_extra_nodes)}, {np.mean(num_extra_nodes)}, {np.max(num_extra_nodes)}, {np.sum(num_orig_nodes)}, {np.sum(num_orig_nodes)**2}, {np.linalg.norm(num_subgraph_nodes)**2}\n")
f.close()
# print(f"##### Coarsening Ratio {args.coarsening_ratio} #####")
# print(f"Tot Num Extra Nodes: {np.sum(num_extra_nodes)}")
# print(f"Avg Num Extra Nodes: {np.mean(num_extra_nodes)}")
# print(f"Max Num Extra Nodes: {np.max(num_extra_nodes)}\n")
# print(num_extra_nodes)