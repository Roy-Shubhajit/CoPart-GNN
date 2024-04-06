import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
import numpy as np
from utils import load_data, coarsening
import os

def train_M1(model, x, edge_index, mask, y, loss_fn, optimizer, metric_fn):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item(), metric_fn(out[mask], y[mask])

def infer_M1(model, x, edge_index, mask, y, loss_fn, metric_fn):
    model.eval()
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return loss.item(), metric_fn(out[mask], y[mask])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
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
    model1 = Net(args).to(device)
    model2 = Net(args).to(device)
    all_acc = []

    for i in range(args.runs):

        data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
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

        best_val_loss = float('inf')
        val_loss_history = []
        for epoch in range(args.epochs):
            train_loss, train_acc = train_M1(model1, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, F.nll_loss, optimizer1, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()))
            val_loss, val_acc = infer_M1(model1, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, F.nll_loss, metric_fn=lambda x, y: int(x.max(1)[1].eq(y).sum().item()) / int(y.sum()))

            if val_loss < best_val_loss and epoch > args.epochs // 2:
                best_val_loss = val_loss
                torch.save(model1.state_dict(), path + 'checkpoint-best-acc.pkl')

            val_loss_history.append(val_loss)
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred = model(data.x, data.edge_index).max(1)[1]
        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
        print(test_acc)
        all_acc.append(test_acc)

    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

