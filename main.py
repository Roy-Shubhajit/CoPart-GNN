import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import Net1
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import load_data, coarsening, create_distribution_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_Gc(model, x, edge_index, mask, y, loss_fn, optimizer):
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

def val_Gc(model, x, edge_index, mask, y, loss_fn):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    mask = mask.to(device)
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return loss.item()

def train_Gs(model, graph_data, loss_fn, optimizer):
    total_loss = 0
    n = 0
    for graph in graph_data:
        train_mask = graph.train_mask.to(device)
        if True in train_mask:
            model.train()
            optimizer.zero_grad()
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            out = model(x, edge_index)
            loss = loss_fn(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            continue
        
    return total_loss / n

def infer_Gs(model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    n = 0
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
                total_time += time.time() - start_time
                test_mask = graph.test_mask.to(device)
                loss = loss_fn(out[test_mask], y[test_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[test_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[test_mask]), dim=0)
            else:
                continue
        else:
            if True in graph.val_mask:
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                val_mask = graph.val_mask.to(device)
                loss = loss_fn(out[val_mask], y[val_mask])
                total_loss += loss.item()
                all_out = torch.cat((all_out, torch.max(out[val_mask], dim=1)[1].to(device)), dim=0)
                all_label = torch.cat((all_label, y[val_mask]), dim=0)
            else:
                continue
        n = n + 1
    
    return total_loss / n, int(all_out.eq(all_label).sum().item()) / int(all_label.shape[0]), total_time

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
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_train') #'Gc_train_2_Gs_train', 'Gc_train_2_Gs_infer', 'Gs_train_2_Gs_infer'
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

    path = "save/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)

    args.num_features, args.num_classes, candidate, C_list, Gc_list, subgraph_list = coarsening(args, 1-args.coarsening_ratio, args.coarsening_method)

    all_acc = []
    all_time = []

    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data(args, args.dataset, candidate, C_list, Gc_list, args.experiment, subgraph_list)
        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
        graph_data = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

        model = Net1(args).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.NLLLoss().to(device)

        if args.exp_setup == 'Gc_train_2_Gs_train':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1)):
                train_loss = train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                run_writer.add_scalar('Gc_train_loss', train_loss, epoch)
                val_loss = val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                run_writer.add_scalar('Gc_val_loss', val_loss, epoch)

                if val_loss < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss
                    torch.save(model.state_dict(), path+'/model.pt') 

            #Train and val on Gs
            model = model.load_state_dict(torch.load(path+'/model.pt'))      
            for epoch in range(args.epochs2):
                train_loss = train_Gs(model, graph_data, loss_fn, optimizer)
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                val_loss, val_acc, val_time = infer_Gs(model, graph_data, loss_fn, 'val')
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
            
            #Test on Gs
            model = model.load_state_dict(torch.load(path+'/model.pt')) 
            test_loss, test_acc, test_time = infer_Gs(model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_acc.append(test_acc)
            all_time.append(test_time)
        
        elif args.exp_setup == 'Gc_train_2_Gs_infer':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1)):
                train_loss = train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                run_writer.add_scalar('Gc_train_loss', train_loss, epoch)
                val_loss = val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                run_writer.add_scalar('Gc_val_loss', val_loss, epoch)

                if val_loss < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss
                    torch.save(model.state_dict(), path+'/model.pt') 

            #Infer on Gs
            model = model.load_state_dict(torch.load(path+'/model.pt'))      
            val_loss, val_acc, val_time = infer_Gs(model, graph_data, loss_fn, 'val')
            run_writer.add_scalar('Gs_val_loss', val_loss, 0)
            run_writer.add_scalar('Gs_val_acc', val_acc, 0)
            test_loss, test_acc, test_time = infer_Gs(model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_acc.append(test_acc)
            all_time.append(test_time)

        elif args.exp_setup == 'Gs_train_2_Gs_infer':
            best_val_loss_Gs =  float('inf')
            #Train on Gs
            for epoch in range(args.epochs2):
                train_loss = train_Gs(model, graph_data, loss_fn, optimizer)
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                val_loss, val_acc, val_time = infer_Gs(model, graph_data, loss_fn, 'val')
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
            
            #Test on Gs
            model = model.load_state_dict(torch.load(path+'/model.pt')) 
            test_loss, test_acc, test_time = infer_Gs(model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_acc.append(test_acc)
            all_time.append(test_time)

    top_acc = sorted(all_acc, reverse=True)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,exp_setup,extra_nodes,cluster_node,hidden,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"exp_setup: {args.exp_setup}")
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