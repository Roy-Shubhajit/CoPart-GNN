import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net1, Net2
import numpy as np
from utils import load_data, coarsening, create_distribution_tensor
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
            new_add = torch.abs((class_dist[i] - torch.sum(out_prob.T[i]))/len(out_prob)).reshape(1).to(device)
            add_tensor = torch.cat((add_tensor, new_add), 0)
        
        return torch.add(F.nll_loss(predictions, targets), torch.sum(add_tensor))

def train_M1(model, x, edge_index, mask, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    out, E_meta = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def infer_M1(model, x, edge_index, mask, y, loss_fn):
    model.eval()
    out, E_meta = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return E_meta, loss.item()

def train_M2(model, graphs, E_meta, loss_fn, optimizer):
    total_loss = 0
    #shuffle graphs to avoid overfitting
    np.random.shuffle(graphs)
    for graph in graphs:
        model.train()
        optimizer.zero_grad()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        E_meta_pass = E_meta[graph.meta_idx].to(device)
        if True in graph.train_mask:
            out = model(x, edge_index, E_meta_pass)
            loss = loss_fn(out[graph.train_mask], y[graph.train_mask])
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        else:
            continue
    return total_loss / len(graphs)

def infer_M2(model, graphs, E_meta, loss_fn, infer_type):
    total_loss = 0
    all_out = torch.tensor([], dtype=torch.float32).to(device)
    all_label = torch.tensor([], dtype=torch.float32).to(device)
    for graph in graphs:
        model.eval()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        E_meta_pass = E_meta[graph.meta_idx].to(device)
        if infer_type == 'test':
            if True in graph.test_mask:
                out = model(x, edge_index, E_meta_pass)
                loss = loss_fn(out[graph.test_mask], y[graph.test_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[graph.test_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[graph.test_mask]), dim=0)
            else:
                continue
        else:
            if True in graph.val_mask:
                out = model(x, edge_index, E_meta_pass)
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
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs1', type=int, default=30)
    parser.add_argument('--epochs2', type=int, default=200)
    parser.add_argument('--num_layers1', type=int, default=2)
    parser.add_argument('--num_layers2', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    path = "save/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)
    args.num_features, args.num_classes, candidate, C_list, Gc_list, map_list = coarsening(args, args.dataset, 1-args.coarsening_ratio, args.coarsening_method)
    print('num_features: {}, num_classes: {}'.format(args.num_features, args.num_classes))
    print('Number of components: {}'.format(len(candidate)))
    model1 = Net1(args).to(device)
    model2 = Net2(args).to(device)
    all_acc = []

    for i in range(args.runs):
        print(f"####################### Run {i+1}/{args.runs} #######################")
        run_writer = SummaryWriter(path+'/run_'+str(i+1))
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_test_labels, coarsen_test_mask, coarsen_edge, graphs = load_data(args, 
            args.dataset, candidate, C_list, Gc_list, args.experiment, map_list)
        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_test_labels = coarsen_test_labels.to(device)
        coarsen_test_mask = coarsen_test_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
        
        graph_data = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)  

        model1.reset_parameters()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model2.reset_parameters()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #accuracy = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
        #new_loss = new_loss_fn(args).to(device)
        new_loss = torch.nn.NLLLoss().to(device)
        best_val_loss_M1 = float('inf')
        best_val_loss_M2 = float('inf')
        val_loss_history_M1 = []
        val_loss_history_M2  = []
        #training Model 1
        for epoch in tqdm(range(args.epochs1), desc='Training Model 1',ascii=True):
            train_loss = train_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_train_mask, y=coarsen_train_labels, loss_fn=F.l1_loss, optimizer=optimizer1)
            E_meta, val_loss = infer_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_val_mask, y=coarsen_val_labels, loss_fn=F.l1_loss)
            #if (epoch+1)%5 == 0 or epoch == 0:
                
            if val_loss < best_val_loss_M1 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                best_val_loss_M1 = val_loss
                torch.save(model1.state_dict(), path + 'checkpoint-best-loss-model-1.pkl')
            val_loss_history_M1.append(val_loss)
            run_writer.add_scalar('Model 1 - Loss/train', train_loss, epoch)
            run_writer.add_scalar('Model 1 - Loss/val', val_loss, epoch)

        
        #training Model 2
        for epoch in tqdm(range(args.epochs2), desc='Training Model 2',ascii=True):
            model1.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-1.pkl'))
            E_meta, val_loss = infer_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_val_mask, y=coarsen_val_labels, loss_fn=F.l1_loss)

            train_loss = train_M2(model=model2, graphs=graphs, E_meta=E_meta, loss_fn=new_loss, optimizer=optimizer2)
            val_loss, val_acc = infer_M2(model=model2, graphs=graphs, E_meta=E_meta, loss_fn=new_loss, infer_type='val')
            #if (epoch+1)%5 == 0 or epoch == 0:
                
            if val_loss < best_val_loss_M2 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs2} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                best_val_loss_M2 = val_loss
                torch.save(model2.state_dict(), path + 'checkpoint-best-loss-model-2.pkl')
            val_loss_history_M2.append(val_loss)
            run_writer.add_scalar('Model 2 - Loss/train', train_loss, epoch)
            run_writer.add_scalar('Model 2 - Loss/val', val_loss, epoch)
            run_writer.add_scalar('Model 2 - Accuracy/val', val_acc, epoch)

        #testing Model 1 and 2
        print("Testing Model 1 and 2")
        model1.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-1.pkl'))
        model2.load_state_dict(torch.load(path + 'checkpoint-best-loss-model-2.pkl'))
        E_meta, test_loss = infer_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_test_mask, y=coarsen_test_labels, loss_fn=F.l1_loss)
        test_loss, test_acc = infer_M2(model=model2, graphs=graphs, E_meta=E_meta, loss_fn=new_loss, infer_type='test')
        writer.add_scalar('Model 2 - Accuracy/test', test_acc, i)
        all_acc.append(test_acc)
        print(f"Run {i+1}/{args.runs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print("#####################################################################")
    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

