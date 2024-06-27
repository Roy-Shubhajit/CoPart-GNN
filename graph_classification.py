import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import Classify_graph_gc, Classify_graph_gs
import torch.nn.functional as F
from torch.utils.data import DataLoader as T_DataLoader
from torch_geometric.data import DataLoader as G_DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import load_data_classification, coarsening_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_Gc(model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for graph in loader:
        graph = graph.to(device)
        out = model(graph.x, graph.edge_index)
        loss = loss_fn(out, graph.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_Gc(model, loader, loss_fn):
    total_loss = 0
    model.eval()
    for graph in loader:
        graph = graph.to(device)
        out = model(graph.x, graph.edge_index)
        loss = loss_fn(out, graph.y)
        total_loss += loss.item()
    return total_loss / len(loader)

def train_Gs(model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for Graphs in loader:
        graphs, batch = Graphs
        y = torch.tensor([]).to(device)
        for graph in graphs:
            graph.to(device)
            y = torch.cat((y, graph.y))
        batch = batch.to(device)
        out = model(graphs, batch)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_Gs(model, loader, loss_fn):
    total_loss = 0
    model.eval()
    for Graphs in loader:
        graphs, batch = Graphs
        y = torch.tensor([]).to(device)
        for graph in graphs:
            graph.to(device)
            y = torch.cat((y, graph.y))
        batch = batch.to(device)
        out = model(graphs, batch)
        loss = loss_fn(out, y)
        total_loss += loss.item()
    return total_loss / len(loader), int(torch.sum(torch.argmax(out, dim=1) == y).item()) / len(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--num_layers1', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs1', type=int, default=100)
    parser.add_argument('--epochs2', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--cluster_node', type=bool, default=False)
    parser.add_argument('--super_graph', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    path = "save/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)
    

