import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import Classify_node
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from new_utils import load_data_classification, load_data_regression, coarsening_classification, coarsening_regression, create_distribution_tensor, adj_matrix_2_edge_index, load_data

from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, QM7b
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add Respective Graph Level Training and Inference Functions 

def process_dataset(args):
    # Node Classification
    if args.task == 'node_cls':
        if args.dataset == 'dblp':
            dataset = CitationFull(root='./dataset', name=args.dataset)
        elif args.dataset == 'Physics':
            dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
        else:
            dataset = Planetoid(root='./dataset', name=args.dataset)
        data = dataset[0]
        # num_classes = len(set(np.array(data.y)))                          ### Might want to Uncomment this to return the number of classes as well 
    elif args.task == 'node_reg':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        data = dataset[0]
    elif args.task == 'graph_cls':
        if args.dataset == 'ENZYMES':
            dataset = TUDataset(root='./dataset', name=args.dataset)
        else:
            dataset = Planetoid(root='./dataset', name=args.dataset)
        # dataset = TUDataset(root='./dataset', name=args.dataset)
        data = dataset
    else:                                                                   
        dataset = TUDataset(root='./dataset', name = args.dataset)          ### Need to add datasets for graph_reg but have temporarily added dataset for graph_cls
        data = dataset
    # if args.normalize_features:
    #     data.x = torch.nn.functional.normalize(data.x, p=1)               ### Attention needed: Normalizing the features
    return data

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
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_infer') # 'Gc_train_2_Gs_infer', 'Gs_train_2_Gs_infer'
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--epochs1', type=int, default=100)
    parser.add_argument('--epochs2', type=int, default=300)
    parser.add_argument('--num_layers1', type=int, default=2)
    parser.add_argument('--num_layers2', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--cluster_node', type=bool, default=False)
    parser.add_argument('--super_graph', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type = str, default = 'node_cls')         ### node_reg, graph_cls, graph_reg
    parser.add_argument('--seed', type = int, default = None)               ### Seed for reproducibility
    args = parser.parse_args()

    args = arg_correction(args)

    path = f"save/{args.task}/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)

    data = process_dataset(args)

    match(args.task):
        case 'node_cls':
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, data, 1-args.coarsening_ratio, args.coarsening_method)
        case 'node_reg':
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, data, 1-args.coarsening_ratio, args.coarsening_method)
        case 'graph_cls':
            Gc_dict = dict()
            Gs_dict = dict()
            y = []
            for i in range(len(data)):
                y.append(data[i].y)
                args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, data[i], 1-args.coarsening_ratio, args.coarsening_method)
                # GC_LIST = []
                # if len(C_list) > 1:
                #     temp = []
                #     for j in range(len(C_list)):
                #         edge_index = torch.tensor(np.array(Gc_list[j].A.nonzero()), dtype=torch.long)
                #         coarsen_features = torch.tensor(C_list[j].toarray() @ data[i].x.numpy()[candidate[j].info["orig_idx"]], dtype=torch.float)
                #         temp.append(Data(x = coarsen_features, edge_index = edge_index, y = data[i].y))
                #     GC_LIST.append(temp)
                # elif len(C_list) == 1:
                #     edge_index = torch.tensor(np.array(Gc_list[0].A.nonzero()), dtype=torch.long)
                #     coarsen_features = torch.tensor(C_list[0].toarray() @ data[i].x.numpy()[candidate[0].info["orig_idx"]], dtype=torch.float)
                #     GC_LIST.append(Data(x = coarsen_features, edge_index = edge_index, y = data[i].y))
                    
                # Gc_dict[i] = GC_LIST if len(C_list) <= 1 else GC_LIST[0]
                # Gs_dict[i] = list(component_2_subgraphs.values()) if len(component_2_subgraphs) > 1 else subgraph_list
        case 'graph_reg':
            Gc_dict = dict()
            Gs_dict = dict()
            y = []
            for i in range(len(data)):
                y.append(data[i].y)
                args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, data[i], 1-args.coarsening_ratio, args.coarsening_method)
                # GC_LIST = []
                # if len(C_list) > 1:
                #     temp = []
                #     for j in range(len(C_list)):
                #         edge_index = torch.tensor(np.array(Gc_list[j].A.nonzero()), dtype=torch.long)
                #         coarsen_features = torch.tensor(C_list[j].toarray() @ data[i].x.numpy()[candidate[j].info["orig_idx"]], dtype=torch.float)
                #         temp.append(Data(x = coarsen_features, edge_index = edge_index, y = data[i].y))
                #     GC_LIST.append(temp)
                # elif len(C_list) == 1:
                #     edge_index = torch.tensor(np.array(Gc_list[0].A.nonzero()), dtype=torch.long)
                #     coarsen_features = torch.tensor(C_list[0].toarray() @ data[i].x.numpy()[candidate[0].info["orig_idx"]], dtype=torch.float)
                #     GC_LIST.append(Data(x = coarsen_features, edge_index = edge_index, y = data[i].y))
                    
                # Gc_dict[i] = GC_LIST if len(C_list) <= 1 else GC_LIST[0]
                # Gs_dict[i] = list(component_2_subgraphs.values()) if len(component_2_subgraphs) > 1 else subgraph_list
    

    match(args.task):
        case 'node_cls':
            coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data_classification(args, data, candidate, C_list, Gc_list, args.experiment, subgraph_list)
        case 'node_reg':
            graphs = load_data_regression(args, data, subgraph_list)
        case 'graph_cls':
            train, val, test = load_data(Gc_dict, Gs_dict, y, args)
        case 'graph_reg':
            train, val, test = load_data(Gc_dict, Gs_dict, y, args)