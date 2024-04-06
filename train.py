import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net1, Net2
import numpy as np
from utils import load_data, coarsening
import os

def train_M1(model, x, edge_index, mask, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    out, E_meta = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def infer_M1(model, x, edge_index, mask, y, loss_fn, metric_fn):
    model.eval()
    out, E_meta = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return E_meta, loss.item(), metric_fn(out[mask], y[mask])

def train_M2(model, graphs, E_meta, loss_fn, optimizer):
    total_loss = 0
    for graph in graphs:
        model.train()
        optimizer.zero_grad()
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        E_meta = E_meta[graph.idx].to(device)
        out = model(x, edge_index, E_meta)
        loss = loss_fn(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(graphs)

def infer_M2(model, graphs, E_meta, loss_fn, metric_fn, infer_type):
    total_loss = 0
    all_out = []
    all_label = []
    for graph in graphs:
        model.eval()
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        E_meta = E_meta[graph.idx].to(device)
        out = model(x, edge_index, E_meta)
        if infer_type == 'test':
            loss = loss_fn(out[graph.test_mask], graph.y[graph.test_mask])
            all_out.append(out[graph.test_mask])
            all_label.append(graph.y[graph.test_mask])
        else:
            loss = loss_fn(out[graph.val_mask], graph.y[graph.val_mask])
            all_out.append(out[graph.val_mask])
            all_label.append(graph.y[graph.val_mask])
        total_loss += loss.item()
    all_out = torch.cat(all_out, dim=0)
    all_label = torch.cat(all_label, dim=0)
    return total_loss / len(graphs), metric_fn(all_out, all_label)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs1', type=int, default=60)
    parser.add_argument('--epochs2', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_features, args.num_classes, candidate, C_list, Gc_list, map_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)
    model1 = Net1(args).to(device)
    model2 = Net2(args).to(device)
    all_acc = []

    for i in range(args.runs):

        data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data(
            args.dataset, candidate, C_list, Gc_list, args.experiment, map_list)
        data = data.to(device)
        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
            data.x = F.normalize(data.x, p=1)

        model1.reset_parameters()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model2.reset_parameters()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss_M1 = float('inf')
        best_val_loss_M2 = float('inf')
        val_loss_history_M1 = []
        val_loss_history_M2  = []
        for epoch in range(args.epochs1):
            train_loss = train_M1(model1, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, F.nll_loss, optimizer1)
            E_meta, val_loss, val_acc = infer_M1(model1, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, F.nll_loss, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()))

            if val_loss < best_val_loss_M1:
                best_val_loss_M1 = val_loss
                torch.save(model1.state_dict(), path + 'checkpoint-best-loss-model-1.pkl')
            val_loss_history_M1.append(val_loss)
            if epoch > args.early_stopping and val_loss_history_M1[-1] > val_loss_history_M1[-args.early_stopping]:
                break
        
        for epoch in range(args.epochs2):
            model1.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-1.pkl'))
            E_meta, val_loss, val_acc = infer_M1(model1, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, F.nll_loss, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()))

            train_loss = train_M2(model2, graphs, E_meta, F.nll_loss, optimizer2)
            val_loss, val_acc = infer_M2(model2, graphs, E_meta, F.nll_loss, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()), infer_type='val')

            if val_loss < best_val_loss_M2:
                best_val_loss_M2 = val_loss
                torch.save(model2.state_dict(), path + 'checkpoint-best-loss-model-2.pkl')
            if epoch > args.early_stopping and val_loss_history_M2[-1] > val_loss_history_M2[-args.early_stopping]:
                break

        model1.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-1.pkl'))
        model2.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-2.pkl'))
        test_loss, test_acc = infer_M2(model2, graphs, E_meta, F.nll_loss, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()), infer_type='test')
        all_acc.append(test_acc)

    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

