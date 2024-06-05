import os
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from utils import load_data, coarsening, create_distribution_tensor
from torch_geometric.data import DataLoader
from pprint import pprint
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_correction(args):
    if args.super_graph:
        args.cluster_node = False
        args.extra_node = False
    elif args.cluster_node:
        args.extra_node = False
        args.super_graph = False
    elif args.extra_node:
        args.cluster_node = False
        args.super_graph = False

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES')
    # parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    # parser.add_argument('--runs', type=int, default=50)
    # parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_train') #'Gc_train_2_Gs_train', 'Gc_train_2_Gs_infer', 'Gs_train_2_Gs_infer'
    # parser.add_argument('--hidden', type=int, default=512)
    # parser.add_argument('--epochs1', type=int, default=100)
    # parser.add_argument('--epochs2', type=int, default=300)
    # parser.add_argument('--num_layers1', type=int, default=2)
    # parser.add_argument('--num_layers2', type=int, default=2)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--extra_node', type=bool, default=True)
    parser.add_argument('--cluster_node', type=bool, default=False)
    parser.add_argument('--super_graph', type=bool, default=False)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    # parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    args = arg_correction(args)

    Gc_dict, Gs_dict, y= coarsening(args, 1-args.coarsening_ratio, args.coarsening_method)
    """
    Outputs:

    Gc_dict -> Dictionary of List of Coarsened Graphs
    Example: Gc_dict[0] returns List of Coarsened Graphs for graph 0 in the dataset
             Gc_dict[1] returns List of Coarsened Graphs for graph 1 in the dataset
    
    Gs_dict -> Dictionary of List of Subgraph List
    Example: Gs_dict[0] returns List of Subgraph List for graph 0 in the dataset
             Gs_dict[0][0] returns Subgraph List for Coarsened Graph 0 in graph 0 in the dataset

    y       -> List of labels for each graph in the dataset
    """
    # for i in [0, 100, 200, 300, 400, 500]:
    #     print(f"x: {len(Gs_dict[i][0][0].x)}")
    #     print(f"y/actual_y:  {Gs_dict[i][0][0].y}/{y[i]}")
    #     print(f"orig_idx: {Gs_dict[i][0][0].orig_idx}")
    #     print(f"Map dict: {Gs_dict[i][0][0].map_dict}")
    #     print(f"Actual ext: {Gs_dict[i][0][0].actual_ext}")

    train, val, test = load_data(Gc_dict, Gs_dict, y, args)

    """
    Parameters:
    args.train_split    ->  Train split ratio wrt whole dataset
    args.val_split      ->  Validation split ratio wrt whole dataset
    args.test_split     ->  Test split ratio wrt whole dataset
    
    NOTE: Sum of all splits must sum up to 1.

    args.seed -> Random seed for splitting the dataset (By default, None, implies every experiment will have random splits)

    Outputs:
    train -> list of [Gc, Gs, mask, y]-type objects. Each object is a training sample.
    Here,   Gc      -> List of Coarsened Graphs for that sample
            Gs      -> List of Subgraph List for that sample
            mask    -> List of orig_idx mask for that sample
            y       -> Label for that sample
    """
    # for i in [0, 1]:
    #     print("##########")
    #     print(f"Gc: {train[i][0]}")
    #     print(f"Gs: {train[i][1]}")
    #     print(f"masks: {train[i][2]}")
    #     print(f"y: {train[i][3]}")

