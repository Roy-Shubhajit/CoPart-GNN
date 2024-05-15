from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_super_graph(dataset, component_2_subgraphs, CLIST, GcLIST):
    # super_graph is the final data object which has combined graphs of all components
    # Not considering masking variables here
    sub_super_graph_list = []
    for component in component_2_subgraphs.keys():
        sub_super_graph = DataLoader(component_2_subgraphs[component], batch_size=len(component_2_subgraphs[component]), shuffle=False)
        for graph in sub_super_graph:
            G_new = Data(x = graph.x, y = graph.y, edge_index = graph.edge_index, ptr = graph.ptr)
            orig_idx_2_sub_super_graph = dict() # USING DICTIONARY FOR ORIGINAL NODE TO SUPER GRAPH NODE MAPPING
            for idx, node in enumerate(graph.orig_idx):
                orig_idx_2_sub_super_graph[node.item()] = idx
            cluster_node_2_sub_super_graph = (G_new.x.shape[0]) + np.arange(G_new.ptr.shape[0]-1) # CLUSTER NODES GIVEN NUMBERS AFTER NORMAL NODE NUMBERS IN SUPER GRAPH
            new_edges = np.array([], dtype=np.compat.long)
            actual_ext = set()
            for node in range(G_new.x.shape[0]):
                N_node_in_orig = neighbour(dataset[0], graph.orig_idx[node].item())
                N_node_in_subgraph = neighbour(G_new, node)
                Nt_node = []
                for n_node in N_node_in_orig:
                    if orig_idx_2_sub_super_graph[n_node] not in N_node_in_subgraph:
                        Nt_node.append(orig_idx_2_sub_super_graph[n_node])
                Nt_node = np.array(Nt_node)
                cluster_nodes = []
                if Nt_node.shape[0] >= 1:
                    for nt_node in Nt_node:
                        cluster_nodes.append(np.argwhere(nt_node >= G_new.ptr.numpy())[-1][0])
                        actual_ext.add(cluster_nodes[-1])
                        e1 = np.array([node, cluster_node_2_sub_super_graph[cluster_nodes[-1]]], dtype=np.compat.long)
                        e2 = np.array([cluster_node_2_sub_super_graph[cluster_nodes[-1]], node], dtype=np.compat.long)
                        if len(new_edges.shape) <= 1:
                            new_edges = np.concatenate((new_edges, e1), axis=0)
                            new_edges = new_edges.reshape(1, 2)
                            new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                        else:
                            new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                            new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)

            # ADDED ALL CROSS EDGES BETWEEN SELECTED CLUSTERS AS GIVEN IN Gc
            actual_ext = list(actual_ext)
            G_new.actual_ext = torch.tensor(np.array(actual_ext) + (G_new.x.shape[0]), dtype = float)
            coar_adj_mat = GcLIST[component].A.toarray()
            for i in range(len(G_new.actual_ext)-1):
                for j in range(i+1, len(G_new.actual_ext)):
                    if coar_adj_mat[actual_ext[i],actual_ext[j]] or coar_adj_mat[actual_ext[j],actual_ext[i]]:
                        e1 = np.array([G_new.actual_ext[i].item(), G_new.actual_ext[j].item()], dtype=np.compat.long)
                        e2 = np.array([G_new.actual_ext[j].item(), G_new.actual_ext[i].item()], dtype=np.compat.long)
                        new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                        new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)

            cluster_features = CLIST[component].dot(G_new.x)
            G_new.x = torch.cat((G_new.x, torch.tensor(cluster_features).float()), dim = 0)
            G_new.y = torch.cat((G_new.y, torch.zeros(len(cluster_features)).long()), dim = 0)
            G_new.edge_index = torch.cat((G_new.edge_index.T, torch.tensor(new_edges, dtype=torch.long)), dim=0).T
            G_new.num_classes = graph.num_classes
            G_new.orig_idx_2_sub_super_graph = orig_idx_2_sub_super_graph
        sub_super_graph_list.append(G_new)

    return sub_super_graph_list

def create_distribution_tensor(input_tensor, class_count):
    if input_tensor.dtype != torch.int64:
        input_tensor = input_tensor.long()
    distribution_tensor = torch.zeros(class_count, dtype=torch.int64).to(device=input_tensor.device)
    unique_classes, counts = input_tensor.unique(return_counts=True)
    distribution_tensor[unique_classes-1] = counts
    return distribution_tensor

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def neighbour(G, node):
    edge_index = G.edge_index.numpy()
    edges_connected_to_k = np.nonzero(edge_index[0] == node)[0]
    neighbors_k = edge_index[1][edges_connected_to_k].flatten()
    return neighbors_k

def nodes_2_neighbours(G, nodes):
    edge_index = G.edge_index.numpy()
    mask = np.isin(edge_index[0], nodes)
    connected_targets = np.unique(edge_index[1, mask])
    return connected_targets

def neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node):
    connected_clusters = np.array([], dtype=int)
    for node in Nt_node:
        comp_node = node_2_comp_node[node]
        meta_node = comp_node_2_meta_node[comp_node]
        connected_clusters = np.append(connected_clusters, meta_node)
    connected_clusters = np.unique(connected_clusters)
    return connected_clusters

def extract_components(H):
        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

def orig_to_new_map(idxs):
    comp_node_2_node, node_2_comp_node = {}, {}
    num = 0
    for i in idxs:
        comp_node_2_node[num] = i
        node_2_comp_node[i] = num
        num += 1
    return comp_node_2_node, node_2_comp_node

def subgraph_mapping(map_dict):
    subgraph_mapping = {}
    for i in map_dict[0].keys():
        new_map = map_dict[0][i]
        if len(map_dict) > 1:
            for j in range(1, len(map_dict)):
                new_map = map_dict[j][new_map]
        subgraph_mapping[i] = new_map
    return subgraph_mapping

def metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node):
    metanode_2_node = {}
    for comp_node, metanode in comp_node_2_meta_node.items():
        if metanode not in metanode_2_node.keys():
            metanode_2_node[metanode] = np.array([comp_node_2_node[comp_node]], dtype=np.compat.long)
        else:
            metanode_2_node[metanode] = np.append(metanode_2_node[metanode], comp_node_2_node[comp_node])
    return metanode_2_node

def coarsening(args, coarsening_ratio, coarsening_method):
    if args.dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=args.dataset)
    elif args.dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
    else:
        dataset = Planetoid(root='./dataset', name=args.dataset)
    data = dataset[0]
    num_classes = len(set(np.array(data.y)))
    if args.normalize_features:
        data.x = torch.nn.functional.normalize(data.x, p=1)
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    CLIST = [] # NEWLY ADDED (WORKS FOR CORA)
    GcLIST = [] # NEWLY ADDED (WORKS FOR CORA)
    subgraph_list=[]
    component_2_subgraphs = {}
    while number < len(candidate):
        H = candidate[number]
        new_subgraph_list = []
        H_feature = data.x[H.info['orig_idx']]
        comp_node_2_node, node_2_comp_node = orig_to_new_map(H.info['orig_idx'])
        if len(H.info['orig_idx']) > 1:
            C, Gc, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            adj = Gc.A
            C_dot_H_feature = C.dot(H_feature)
            CLIST.append(C)
            GcLIST.append(Gc)
            if len(H.info['orig_idx']) > 10:
                C_list.append(C)
                Gc_list.append(Gc)
            comp_node_2_meta_node = subgraph_mapping(mapping_dict_list)
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            for key, value in meta_node_2_node.items():
                value = np.sort(value)
                node_2_subgraph_node = {v.item(): i for i, v in enumerate(value)}
                actual_ext = np.array([], dtype=np.compat.long)
                num_nodes = len(value)
                if args.cluster_node:
                    new_edges = np.array([], dtype=np.compat.long)
                    new_features = np.array([])
                    meta_node_2_new_node = {}
                    for node in value:
                        N_node = neighbour(dataset[0], node)
                        Nt_node = N_node[~np.isin(N_node, value)]
                        connected_clusters = neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node)
                        for cluster in connected_clusters:
                            if cluster not in meta_node_2_new_node.keys():
                                meta_node_2_new_node[cluster] = np.array([num_nodes])
                                if len(actual_ext.shape) <= 1:
                                    actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster]), axis=0)
                                    actual_ext = actual_ext.reshape(1, 1)
                                else:
                                    actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster].reshape(1,1)), axis=0)
                                new_feature = C_dot_H_feature[cluster]
                                if len(new_features.shape) <= 1:
                                    new_features = new_feature
                                    new_features = new_features.reshape(1, len(new_feature))
                                else:
                                    new_features = np.concatenate((new_features, new_feature.reshape(1,-1)), axis=0)
                                num_nodes += 1
                            # print(meta_node_2_new_node[cluster] ,np.array([node, meta_node_2_new_node[cluster][0]], dtype=np.compat.long))
                            e1 = np.array([node_2_subgraph_node[node], meta_node_2_new_node[cluster][0]], dtype=np.compat.long)
                            e2 = np.array([meta_node_2_new_node[cluster][0], node_2_subgraph_node[node]], dtype=np.compat.long)
                            if len(new_edges.shape) <= 1:
                                new_edges = np.concatenate((new_edges, e1), axis=0)
                                new_edges = new_edges.reshape(1, 2)
                                new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                            else:
                                new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                        
                    if len(meta_node_2_new_node.keys()) > 1:
                        cluster_keys = list(meta_node_2_new_node.keys())
                        for i in range(len(cluster_keys)-1):
                            for j in range(i+1, len(cluster_keys)):
                                if adj[cluster_keys[i], cluster_keys[j]] or adj[cluster_keys[j], cluster_keys[i]]:
                                    e1 = np.array([meta_node_2_new_node[cluster_keys[i]][0], meta_node_2_new_node[cluster_keys[j]][0]], dtype=np.compat.long)
                                    e2 = np.array([meta_node_2_new_node[cluster_keys[j]][0], meta_node_2_new_node[cluster_keys[i]][0]], dtype=np.compat.long)
                                    new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)

                elif args.extra_node:
                    extra_node = nodes_2_neighbours(dataset[0], value)
                    actual_ext = extra_node[~np.isin(extra_node, value)]
                    value = np.concatenate((value, extra_node), 0)
                    value = np.unique(value)
                
                value = np.sort(value)
                value = torch.tensor(value)
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M = data.subgraph(value)
                M.num_classes = num_classes
                M.actual_ext = actual_ext
                M.orig_idx = value
                new_subgraph_list.append(M)
                if args.cluster_node:
                    M.x = torch.cat((M.x, torch.tensor(new_features).float()), dim=0)
                    M.edge_index = torch.cat((M.edge_index.T, torch.tensor(new_edges, dtype=torch.long)), dim=0).T
                    M.y = torch.cat((M.y, torch.zeros(len(new_features)).long()))
                    for new_node in actual_ext:
                        mappiing[new_node.item()] = new_node.item()
                M.map_dict = mappiing
                subgraph_list.append(M)
        else:
            comp_node_2_meta_node = subgraph_mapping([{0: 0}])
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, node_2_comp_node)
            for key, value in meta_node_2_node.items():
                value = torch.LongTensor(value)
                value, _ = torch.sort(value)
                actual_ext = torch.LongTensor([])
                M = data.subgraph(value)
                M.num_classes = num_classes
                M.actual_ext = actual_ext
                M.orig_idx = value
                new_subgraph_list.append(M)
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M.map_dict = mappiing
                subgraph_list.append(M)
        component_2_subgraphs[number] = new_subgraph_list
        number += 1

    #print("Subgraphs created, number of subgraphs: ", len(subgraph_list))

    if args.super_graph:
        return data.x.shape[1], num_classes, candidate, C_list, Gc_list, (dataset, component_2_subgraphs, CLIST, GcLIST)
    else:
        return data.x.shape[1], num_classes, candidate, C_list, Gc_list, subgraph_list

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

def load_data(args, dataset, candidate, C_list, Gc_list, exp, subgraph_list):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
    n_classes = len(set(np.array(dataset[0].y)))
    data = splits(dataset[0], n_classes, exp)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    new_graphs = []

    if args.super_graph:
        for graph in subgraph_list:
            F = Data(x=graph.x, edge_index=graph.edge_index, y=graph.y, num_classes=graph.num_classes)
            F.train_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            F.val_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            F.test_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            for node, new_node in graph.orig_idx_2_sub_super_graph.items():
                if train_mask[node]:
                    F.train_mask[new_node] = True
                if val_mask[node]:
                    F.val_mask[new_node] = True
                if test_mask[node]:
                    F.test_mask[new_node] = True
                if new_node in graph.actual_ext:
                    F.train_mask[new_node] = False
                    F.val_mask[new_node] = False
                    F.test_mask[new_node] = False
            new_graphs.append(F)

    else:
        for graph in subgraph_list:
            F = Data(x=graph.x, edge_index=graph.edge_index, y=graph.y, num_classes=graph.num_classes)
            F.train_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            F.val_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            F.test_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
            for node, new_node in graph.map_dict.items():
                if train_mask[node]:
                    F.train_mask[new_node] = True
                if val_mask[node]:
                    F.val_mask[new_node] = True
                if test_mask[node]:
                    F.test_mask[new_node] = True
                if args.extra_node and node in graph.actual_ext:
                    F.train_mask[new_node] = False
                    F.val_mask[new_node] = False
                    F.test_mask[new_node] = False
                if args.cluster_node and new_node in graph.actual_ext:
                    F.train_mask[new_node] = False
                    F.val_mask[new_node] = False
                    F.test_mask[new_node] = False
            new_graphs.append(F)

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]

        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            #coarsen_train_labels = torch.cat([coarsen_train_labels, torch.FloatTensor(C.dot(train_labels))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            #coarsen_val_labels = torch.cat([coarsen_val_labels, torch.FloatTensor(C.dot(val_labels))], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            #H_labels = one_hot(H_labels, n_classes)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    #print('the size of coarsen graph features:', coarsen_features.shape)
    coarsen_edge = np.array([coarsen_row, coarsen_col])
    coarsen_edge = torch.LongTensor(coarsen_edge)
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, new_graphs
