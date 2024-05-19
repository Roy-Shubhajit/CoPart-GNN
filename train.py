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
            new_add = (torch.pow((class_dist[i] - torch.sum(out_prob.T[i])), 2)/len(out_prob)).reshape(1).to(device)
            add_tensor = torch.cat((add_tensor, new_add), 0)
        
        return torch.add(F.nll_loss(predictions, targets), torch.sum(add_tensor))

def train_M1(model, x, edge_index, mask, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    mask = mask.to(device)
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def infer_M1(model, x, edge_index, mask, y, loss_fn):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    mask = mask.to(device)
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return loss.item()

def train_M2(model, graph_data, loss_fn, optimizer):
    total_loss = 0
    n = 0
    for graph in graph_data:
        if True in graph.train_mask:
            model.train()
            optimizer.zero_grad()
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            train_mask = graph.train_mask.to(device)
            out = model(x, edge_index)
            loss = loss_fn(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            continue
        n = n + 1
    return total_loss / n

def infer_M2(model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    all_out = torch.tensor([], dtype=torch.float32).to(device)
    all_label = torch.tensor([], dtype=torch.float32).to(device)
    for graph in graph_data:
        model.eval()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        if infer_type == 'test':
            if True in graph.test_mask:
                start_time = time.time()
                out = model(x, edge_index)
                test_mask = graph.test_mask.to(device)
                total_time += time.time() - start_time
                loss = loss_fn(out[test_mask], y[test_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[test_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[test_mask]), dim=0)
            else:
                continue
        else:
            if True in graph.val_mask:
                out = model(x, edge_index)
                val_mask = graph.val_mask.to(device)
                loss = loss_fn(out[val_mask], y[val_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[val_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[val_mask]), dim=0)
            else:
                continue
    
    return total_loss / len(graphs), int(all_out.eq(all_label).sum().item()) / int(all_label.shape[0]), total_time
        
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

if __name__ == '__main__':
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
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--cluster_node', type=bool, default=False)
    parser.add_argument('--super_graph', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    args = arg_correction(args)
    
    print(args)
    path = "save/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)
    args.num_features, args.num_classes, candidate, C_list, Gc_list, subgraph_list = coarsening(args, 1-args.coarsening_ratio, args.coarsening_method)
    #print('num_features: {}, num_classes: {}'.format(args.num_features, args.num_classes))
    #print('Number of components: {}'.format(len(candidate)))
    all_acc = []
    all_time = []
    for i in range(args.runs):
        #print(f"####################### Run {i+1}/{args.runs} #######################")
        run_writer = SummaryWriter(path+'/run_'+str(i+1))
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data(args, args.dataset, candidate, C_list, Gc_list, args.experiment, subgraph_list)
        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
        
        #graph_data = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)  
        graph_data = DataLoader(graphs, batch_size=len(graphs), shuffle=False) 

        model1 = Net1(args).to(device)
        model1.reset_parameters()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #accuracy = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
        #new_loss = new_loss_fn(args).to(device)
        new_loss = torch.nn.NLLLoss().to(device)
        best_val_loss_M1 = float('inf')
        best_val_loss_M2 = float('inf')
        best_val_acc_M2 = 0
        val_loss_history_M1 = []
        val_loss_history_M2  = []
        #training Model 1
        #for epoch in tqdm(range(args.epochs1), desc='Training Model 1',ascii=True):
        for epoch in range(args.epochs1):
            train_loss = train_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_train_mask, y=coarsen_train_labels, loss_fn=F.nll_loss, optimizer=optimizer1)
            val_loss = infer_M1(model=model1, x=coarsen_features, edge_index=coarsen_edge, mask=coarsen_val_mask, y=coarsen_val_labels, loss_fn=F.nll_loss)
            #if (epoch+1)%5 == 0 or epoch == 0:
                #print(f"Epoch {epoch+1}/{args.epochs1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss_M1 or epoch == 0:
                #print(f"Epoch {epoch+1}/{args.epochs1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                best_val_loss_M1 = val_loss
                torch.save(model1.state_dict(), path+'/model1.pt')
            val_loss_history_M1.append(val_loss)
            run_writer.add_scalar('Model 1 - Loss/train', train_loss, epoch)
            run_writer.add_scalar('Model 1 - Loss/val', val_loss, epoch)

        model1.load_state_dict(torch.load(path+'/model1.pt'))
        #for param in model1.conv.parameters():
            #param.requires_grad = False

        #model2 = TransferNet(args, model1).to(device)
        #model2.reset_parameters()
        #optimizer2 = torch.optim.Adam(model2.new_lt1.parameters(), lr=0.003, weight_decay=args.weight_decay)

        #training Model 2
        #for epoch in tqdm(range(args.epochs2), desc='Training Model 2',ascii=True):
        for epoch in range(args.epochs2):
            train_loss = train_M2(model1, graph_data, new_loss, optimizer1)
            val_loss, val_acc, val_time = infer_M2(model1, graph_data, new_loss, 'val')
            #if (epoch+1)%5 == 0 or epoch == 0:
                #print(f"Epoch {epoch+1}/{args.epochs2} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            if val_loss < best_val_loss_M2 or epoch == 0:
                #print(f"Epoch {epoch+1}/{args.epochs2} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                best_val_loss_M2 = val_loss
                torch.save(model1.state_dict(), path+'/model2.pt')
            val_loss_history_M2.append(val_loss)
            run_writer.add_scalar('Model 2 - Loss/train', train_loss, epoch)
            run_writer.add_scalar('Model 2 - Loss/val', val_loss, epoch)
            run_writer.add_scalar('Model 2 - Accuracy/val', val_acc, epoch)

        best_model2 = model1.load_state_dict(torch.load(path+'/model2.pt'))
        test_loss, test_acc, test_time = infer_M2(model1, graph_data, new_loss, 'test')
        writer.add_scalar('Model 2 - Accuracy/test', test_acc, i)
        all_acc.append(test_acc)
        all_time.append(test_time)
        #print(f"Run {i+1}/{args.runs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Time: {test_time:.4f} sec")
        #print("#####################################################################")
    #print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
    #print('ave_time: {:.4f}'.format(np.mean(all_time)), '+/- {:.4f}'.format(np.std(all_time)))
    #print mean and std of top 10 runs
    top_acc = sorted(all_acc, reverse=True)[:10]
    #print('top_10_ave_acc: {:.4f}'.format(np.mean(top_acc)), '+/- {:.4f}'.format(np.std(top_acc)))

    #check whether f"results/{args.dataset}.csv" is present or not, if not create and write a line
    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,extra_nodes,cluster_node,hidden,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc\n')
    #write the results to the csv file
    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.extra_node},{args.cluster_node},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"extra_nodes: {args.extra_node}")
    print(f"cluster_node: {args.cluster_node}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"coarsening_ratio: {args.coarsening_ratio}")
    print(f"coarsening_method: {args.coarsening_method}")
    print(f"ave_acc: {np.mean(all_acc)} +/- {np.std(all_acc)}")
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_acc: {np.mean(top_acc)} +/- {np.std(top_acc)}")
    print("#####################################################################")


