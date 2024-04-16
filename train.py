import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net1, TransferNet
import numpy as np
from utils import load_data, coarsening, create_distribution_tensor
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class new_loss_fn(torch.nn.Module):
    def __init__(self, args):
        super(new_loss_fn, self).__init__()
        self.num_classes = args.num_classes

    def forward(self, predictions, targets):
        add_tensor = torch.tensor([], dtype=torch.float32).to(device)
        out_prob = torch.exp(predictions).to(device)
        class_dist = create_distribution_tensor(targets, self.num_classes).to(device)
        for i in range(len(class_dist)):
            new_add = (torch.pow((class_dist[i] - torch.sum(out_prob.T[i])), 2)/len(out_prob)).reshape(1).to(device)
            add_tensor = torch.cat((add_tensor, new_add), 0)
        
        return torch.add(F.nll_loss(predictions, targets), torch.sum(add_tensor))

def train_M1(model, x, edge_index, mask, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def infer_M1(model, x, edge_index, mask, y, loss_fn):
    model.eval()
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return loss.item()

def train_M2(model, graph_data, loss_fn, optimizer):
    total_loss = 0
    for graph in graph_data:
        model.train()
        optimizer.zero_grad()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        out = model(x, edge_index)
        if True in graph.train_mask:
            loss = loss_fn(out[graph.train_mask], y[graph.train_mask])
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        else:
            continue
    return total_loss / len(graphs)

def infer_M2(model, graph_data, loss_fn, infer_type):
    total_loss = 0
    all_out = torch.tensor([], dtype=torch.float32).to(device)
    all_label = torch.tensor([], dtype=torch.float32).to(device)
    for graph in graph_data:
        model.eval()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        out = model(x, edge_index)
        if infer_type == 'test':
            if True in graph.test_mask:
                loss = loss_fn(out[graph.test_mask], y[graph.test_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[graph.test_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[graph.test_mask]), dim=0)
            else:
                continue
        else:
            if True in graph.val_mask:
                loss = loss_fn(out[graph.val_mask], y[graph.val_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[graph.val_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[graph.val_mask]), dim=0)
            else:
                continue
    
    return total_loss / len(graphs), int(all_out.eq(all_label).sum().item()) / int(all_label.shape[0])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--epochs1', type=int, default=100)
    parser.add_argument('--epochs2', type=int, default=200)
    parser.add_argument('--num_layers1', type=int, default=2)
    parser.add_argument('--num_layers2', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    path = "save/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    args.num_features, args.num_classes, candidate, C_list, Gc_list, subgraph_list = coarsening(args, 1-args.coarsening_ratio, args.coarsening_method)
    print('num_features: {}, num_classes: {}'.format(args.num_features, args.num_classes))
    print('Number of components: {}'.format(len(candidate)))
    all_acc = []

    for i in range(args.runs):
        print(f"####################### Run {i+1}/{args.runs} #######################")
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data(args.dataset, candidate, C_list, Gc_list, args.experiment, subgraph_list)
        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
            for graph in graphs:
                graph.x = F.normalize(graph.x, p=1)
        
        graph_data = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)  

        model1 = Net1(args).to(device)
        model1.reset_parameters()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #accuracy = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
        new_loss = new_loss_fn(args).to(device)
        #new_loss = torch.nn.NLLLoss().to(device)
        best_val_loss_M1 = float('inf')
        best_val_loss_M2 = float('inf')
        val_loss_history_M1 = []
        val_loss_history_M2  = []
        #training Model 1
        for epoch in tqdm(range(args.epochs1), desc='Training Model 1',ascii=True):
            train_loss = train_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_train_mask, y=coarsen_train_labels, loss_fn=F.l1_loss, optimizer=optimizer1)
            val_loss = infer_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_val_mask, y=coarsen_val_labels, loss_fn=F.l1_loss)
            if (epoch+1)%5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss_M1 or epoch == 0:
                best_val_loss_M1 = val_loss
                torch.save(model1.state_dict(), path+'/model1.pt')
            val_loss_history_M1.append(val_loss)

        model1.load_state_dict(torch.load(path+'/model1.pt'))
        for param in model1.conv.parameters():
            param.requires_grad = False

        model2 = TransferNet(args, model1).to(device)
        optimizer2 = torch.optim.Adam(model2.new_lt1.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        #training Model 2
        for epoch in tqdm(range(args.epochs2), desc='Training Model 2',ascii=True):
            train_loss = train_M2(model2, graph_data, new_loss, optimizer2)
            val_loss, val_acc = infer_M2(model2, graph_data, new_loss, 'val')
            if (epoch+1)%5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs2} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            if val_loss < best_val_loss_M2 or epoch == 0:
                best_val_loss_M2 = val_loss
                torch.save(model2.state_dict(), path+'/model2.pt')
            val_loss_history_M2.append(val_loss)

        best_model2 = model2.load_state_dict(torch.load(path+'/model2.pt'))
        test_loss, test_acc = infer_M2(model2, graph_data, new_loss, 'test')
        all_acc.append(test_acc)
        print(f"Run {i+1}/{args.runs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print("#####################################################################")
    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

