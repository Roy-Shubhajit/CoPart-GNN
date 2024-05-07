from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    edges_connected_to_k = torch.nonzero(G.edge_index[0] == node, as_tuple=False)
    neighbors_k = G.edge_index[1][edges_connected_to_k].flatten()
    return neighbors_k

def nodes_2_neighbours(G, nodes):
    if not hasattr(G, 'edge_index'):
        raise ValueError("The input graph must have the 'edge_index' attribute.")
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes)
    nodes = nodes.to(device)
    edge_index_gpu = G.edge_index.to(nodes.device)
    mask = (edge_index_gpu[0] == nodes.unsqueeze(-1)).any(dim=0)
    edges_connected_to_nodes = edge_index_gpu[:, mask]
    neighbors = torch.unique(edges_connected_to_nodes[1])
    return neighbors

def neighbor_2_cluster(Nt_node,node_2_comp_node, comp_node_2_meta_node):
    connected_clusters = torch.tensor([], dtype=torch.long)
    for node in Nt_node:
        comp_node = node_2_comp_node[node]
        meta_node = comp_node_2_meta_node[comp_node]
        connected_clusters = torch.cat((connected_clusters, torch.tensor([meta_node])))
    connected_clusters = torch.unique(connected_clusters)
    return connected_clusters.tolist()

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

def metanode_to_node_mapping_new(map_dict, orig_dict):
    temp = dict()
    for node, metanode in map_dict.items():
        if metanode not in set(temp.keys()):
            temp[metanode] = [orig_dict[node]]
        else:
            temp[metanode].append(orig_dict[node])
    return temp

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
    data.to(device)
    number = 0
    C_list=[]
    Gc_list=[]
    subgraph_list=[]
    while number < len(candidate):
        H = candidate[number]
        H_feature = data.x[H.info['orig_idx']]
        comp_node_2_node, node_2_comp_node = orig_to_new_map(H.info['orig_idx'])
        if len(H.info['orig_idx']) > 1:
            C, Gc, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            if len(H.info['orig_idx']) > 10:
                C_list.append(C)
                Gc_list.append(Gc)
            comp_node_2_meta_node = subgraph_mapping(mapping_dict_list)
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            for key, value in meta_node_2_node.items():
                value = torch.LongTensor(value).to(device)
                value, _ = torch.sort(value)
                actual_ext = torch.LongTensor([]).to(device)
                num_nodes = len(value)
                if args.cluster_node:
                    new_edges = torch.tensor([], dtype=torch.long).to(device)
                    new_features = torch.tensor([], dtype=torch.float).to(device)
                    meta_node_2_new_node = {}
                    for node in value:
                        N_node = neighbour(data, node)
                        Nt_node = N_node[~torch.isin(N_node, value)]
                        connected_clusters = neighbor_2_cluster(Nt_node.tolist(), node_2_comp_node, comp_node_2_meta_node)
                        for cluster in connected_clusters:
                            if cluster not in meta_node_2_new_node:
                                meta_node_2_new_node[cluster] = torch.LongTensor([num_nodes])
                                actual_ext = torch.cat((actual_ext, torch.tensor([num_nodes]).to(device)), dim=0)
                                new_feature = torch.FloatTensor(np.array([C.dot(H_feature.cpu())[cluster]])).to(device)
                                new_features = torch.cat((new_features, new_feature), dim=0)
                                num_nodes += 1
                            e1 = torch.tensor([node, meta_node_2_new_node[cluster]]).to(device)
                            e2 = torch.tensor([meta_node_2_new_node[cluster], node]).to(device)
                            new_edges = torch.cat((new_edges, e1.unsqueeze(0)), dim=0)
                            new_edges = torch.cat((new_edges, e2.unsqueeze(0)), dim=0)
                        #we need to have get the cross cluster edges
                        edge_arr = np.vstack([Gc.get_edge_list()[0], Gc.get_edge_list()[1]]).T.tolist()
                        for i in range(len(connected_clusters)):
                            for j in range(len(connected_clusters)):
                                if i>=j:
                                    if [connected_clusters[i], connected_clusters[j]] in edge_arr or [connected_clusters[j], connected_clusters[i]] in edge_arr:
                                        e1 = torch.tensor([meta_node_2_new_node[connected_clusters[i]], meta_node_2_new_node[connected_clusters[j]]]).to(device)
                                        e2 = torch.tensor([meta_node_2_new_node[connected_clusters[j]], meta_node_2_new_node[connected_clusters[i]]]).to(device)
                                        new_edges = torch.cat((new_edges, e1.unsqueeze(0)), dim=0)
                                        new_edges = torch.cat((new_edges, e2.unsqueeze(0)), dim=0)
                        
                elif args.extra_node:
                    ext_nodes = nodes_2_neighbours(data, value)
                    actual_ext = ext_nodes[~torch.isin(ext_nodes, value)]
                    value = torch.cat((value, ext_nodes), dim=0)
                    value = torch.unique(value)
                value, _ = torch.sort(value)
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M = data.subgraph(value)
                if args.cluster_node:
                    M.x = torch.cat((M.x, new_features), dim=0)
                    M.edge_index = torch.cat((M.edge_index.T, new_edges), dim=0).T
                    M.y = torch.cat((M.y, torch.zeros(len(new_features)).to(device)), dim=0)
                    for new_node in actual_ext:
                        mappiing[new_node.item()] = new_node.item()
                M.num_classes = num_classes
                M.map_dict = mappiing
                M.actual_ext = actual_ext
                subgraph_list.append(M)
        else:
            comp_node_2_meta_node = subgraph_mapping([{0: 0}])
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, node_2_comp_node)
            for key, value in meta_node_2_node.items():
                value = torch.LongTensor(value)
                value, _ = torch.sort(value)
                actual_ext = torch.LongTensor([])
                num_nodes = len(value)
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M = data.subgraph(value)
                M.num_classes = num_classes
                M.map_dict = mappiing
                M.actual_ext = actual_ext
                subgraph_list.append(M)
            number += 1
    #print("Subgraphs created, number of subgraphs: ", len(subgraph_list))
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

    for graph in subgraph_list:
        F = Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)
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
            if args.ext_node and node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
            if args.cluster_node and new_node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
        
        new_graphs.append(F)
    
    torch.cuda.empty_cache()

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
