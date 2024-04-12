from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

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
    neighbors_k = G.edge_index[1][edges_connected_to_k].flatten().tolist()
    return neighbors_k

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
    new_idxs = {}
    num = 0
    for i in idxs:
        new_idxs[num] = i
        num += 1
    return new_idxs

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

def coarsening(dataset, coarsening_ratio, coarsening_method):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
    elif dataset == 'KarateClub':
        dataset = KarateClub()
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
    data = dataset[0]
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    map_list=[]
    while number < len(candidate):
        H = candidate[number]
        original_map = orig_to_new_map(H.info['orig_idx'])
        if len(H.info['orig_idx']) > 1:
            C, Gc, Call, Gall, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
            map_list.append(metanode_to_node_mapping_new(subgraph_mapping(mapping_dict_list), original_map))
        else:
            C = sp.sparse.eye(H.N, format="csc")
            C_list.append(C)
            Gc_list.append(H)
            map_list.append(metanode_to_node_mapping_new(subgraph_mapping([{0: 0}]), original_map))
        number += 1
    inv_map_list = metanode_to_node_mapping(map_list)
    # print(inv_map_list)
    return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list, map_list, inv_map_list, create_data(inv_map_list, data, Gc_list)

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

def load_data(args, dataset, candidate, C_list, Gc_list, exp, map_list):
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
    coarsen_test_labels = torch.Tensor([])
    coarsen_test_mask = torch.Tensor([]).bool()

    subgraph_list = []

    while number < len(candidate):
        H = candidate[number]
        C = C_list[number]
        Gc = Gc_list[number]
        mapping_dict = map_list[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        H_test_mask = test_mask[keep]

        for key, value in mapping_dict.items():
            ext_nodes = []
            if args.extra_node:
                for node in value:
                    ext_nodes.extend(neighbour(data, node))
                ext_nodes = list(set(ext_nodes))
                value.extend(ext_nodes)
                value = list(set(value))
            value = torch.LongTensor(value)
            M = data.subgraph(index_to_mask(value, data.num_nodes))
            M.meta_idx = key+coarsen_node
            M.num_classes = n_classes
            subgraph_map_dict = {int(v): i for i, v in enumerate(value)}
            for node, new_node in subgraph_map_dict.items():
                if node in ext_nodes:
                    M.train_mask[new_node] = False
                    M.val_mask[new_node] = False
                    M.test_mask[new_node] = False
            subgraph_list.append(M)

        #if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
        if torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            test_labels = one_hot(H_labels, n_classes)
            test_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])

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

            new_test_mask = torch.BoolTensor(np.sum(C.dot(test_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(test_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_test_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            #coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.FloatTensor(C.dot(train_labels))], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            #coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0) 
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.FloatTensor(C.dot(val_labels))], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)
            coarsen_test_labels = torch.cat([coarsen_test_labels, torch.FloatTensor(C.dot(test_labels))], dim=0)
            coarsen_test_mask = torch.cat([coarsen_test_mask, new_test_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.N

        #elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0: #Maybe we need to change this statement to else only
        else:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, one_hot(H_labels, n_classes).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, one_hot(H_labels, n_classes).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)
            coarsen_test_labels = torch.cat([coarsen_test_labels, one_hot(H_labels, n_classes).float()], dim=0)
            coarsen_test_mask = torch.cat([coarsen_test_mask, H_test_mask], dim=0)

            if coarsen_row is None:
                #raise Exception('The graph does not need coarsening.')
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.N

        number += 1

    print('the size of coarsen graph features:', coarsen_features.shape)
    coarsen_edge = np.array([coarsen_row, coarsen_col])
    coarsen_edge = torch.LongTensor(coarsen_edge)
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()
    coarsen_test_labels = coarsen_test_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_test_labels, coarsen_test_mask, coarsen_edge, subgraph_list
