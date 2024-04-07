from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull, KarateClub
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import from_scipy_sparse_matrix

def create_distribution_tensor(input_tensor, class_count):
    if input_tensor.dtype != torch.int64:
        input_tensor = input_tensor.long()
    distribution_tensor = torch.zeros(class_count, dtype=torch.int64)
    unique_classes, counts = input_tensor.unique(return_counts=True)
    distribution_tensor[unique_classes-1] = counts
    return distribution_tensor

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

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

def subgraph_mapping(map_dict):
    subgraph_mapping = {}
    for i in map_dict[0].keys():
        new_map = map_dict[0][i]
        if len(map_dict) > 1:
            for j in range(1, len(map_dict)):
                new_map = map_dict[j][new_map]    
        subgraph_mapping[i] = new_map
    return subgraph_mapping

def metanode_to_node_mapping(map_list:list) -> list:
    '''
    Returns a list of dictionaries (each dictionary corresponds to a distinct disconnected component in original graph) containing (`metanode` [key] : list(`node`) [values]) pairs.
    It is essentially an inverse mapping of `map_list`.
    ---
    Parameters
    ----------
    map_list :
        List of dictionaries (each dictionary corresponds to a distinct disconnected component in original graph) containing (`node` [key] : `metanode` [value]) pairs.
    
    Returns
    -------
    inv_map_list :
        Inverse mapping of map_list.
    '''
    inv_map_list = []
    for subgraph in map_list:
        temp = dict()
        for node, metanode in subgraph.items():
            if metanode not in set(temp.keys()):
                temp[metanode] = [node]
            else:
                temp[metanode].append(node)
        inv_map_list.append(temp)
    return inv_map_list

def create_data(inv_map_list:list, data:Data, Gc_list:list) -> list:
    '''
    Returns a list of `torch_geometric.data.Data` type objects (each object corresponds to the coarsened version of distinct disconnected component in original graph)
    containing `edge_index` between metanodes, `node_features` and `node_labels`.
    ---
    Parameters
    ----------
    inv_map_list :
        Inverse mapping of map_list.
    
    data :
        `torch_geometric.data.Data` type object of the original graph. Must contain `x` and `y` as attributes.
    
    Gc_list :
        List of `gsp.graphs.Graph` type objects. Each object is a coarsened version of a `component` of the original graph.
   
    Returns
    -------
    data_list :
        List of `torch_geometric.data.Data` type objects (each object corresponds to the coarsened version of distinct disconnected component in original graph)
        Each object has the following attributes:

        `edge_index` :
                    `torch.tensor` type object containing the undirected edges between metanodes for the particular graph object.
        `node_features` :
                List of metanode features for the particular graph object. Each entry is the features of those node of the original graph which are mapped to the particular metanode.
                For example, if nodes [5, 6, 16] are mapped to metanode 2, i.e., 2: [5, 6, 16], then `data_list.node_features[2]` would be the ordered combined features of nodes [5, 6, 16]\n
                of shape (3 x f), where f is the number of features of any node in original graph.
        `node_labels` :
                List of metanode labels for the particular graph object. Each entry is the labels of those node of the original graph which are mapped to the particular metanode.
                For example, if nodes [5, 6, 16] are mapped to metanode 2, i.e., 2: [5, 6, 16], then `data_list.node_labels[2]` would be the ordered combined labels of nodes [5, 6, 16]\n
                of shape (3 x g), where g is the size of the vector used to define the label of any node in original graph.
    '''
    # print(f"Feature Matrix of original graph:\n{data.x}\n")
    data_list = []
    for idx, component in enumerate(inv_map_list):
        subgraph = Data(edge_index = from_scipy_sparse_matrix(Gc_list[idx].W)[0])
        x = []
        y = []
        # print(f"Subgraph {idx}: {component}\n")
        for indices in component.values():
            x.append(data.x[indices])
            y.append(data.y[indices])
        subgraph.node_features, subgraph.node_labels = x, y
        # print(f"X: {subgraph.node_features}")
        # print(f"Y: {subgraph.node_labels}")
        data_list.append(subgraph)
    return data_list

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
    print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    map_list=[]
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
            map_list.append(subgraph_mapping(mapping_dict_list))

            # #######
            # print(f"Length of Gall: {len(Gall)}\n")
            # for level, graph in enumerate(Gall):
            #     plt.figure()
            #     nx_graph = nx.Graph(graph.W.toarray())
            #     print(graph.W.toarray(), graph.W.toarray().shape)
            #     print(from_scipy_sparse_matrix(graph.W)[0], from_scipy_sparse_matrix(graph.W)[0].shape)
            #     pos = nx.spring_layout(nx_graph)
            #     nx.draw(nx_graph, pos, with_labels=True)
            #     plt.axis('equal')
            #     plt.title(f"Subgraph {number} Level {level}")
            #     plt.show()
            
            # plt.figure()
            # nx_graph = nx.Graph(Gc.W.toarray())
            # print(Gc.W.toarray(), Gc.W.toarray().shape)
            # print(from_scipy_sparse_matrix(Gc.W)[0], from_scipy_sparse_matrix(Gc.W)[0].shape)
            # pos = nx.spring_layout(nx_graph)
            # nx.draw(nx_graph, pos, with_labels=True)
            # plt.axis('equal')
            # plt.title(f"Gc")
            # plt.show()
            # #######

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

def load_data(dataset, candidate, C_list, Gc_list, exp, map_list):
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

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        ###########################
        #create the subgraphs for each H here using map_list
        ###########################
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
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0) #we need to replace labels here to n_class dimensional vector
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0) #we need to replace labels here to n_class dimensional vector
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

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0: #Maybe we need to change this statement to else only

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0) #we need to replace labels here to n_class dimensional vector
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0) #we need to replace labels here to n_class dimensional vector
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

    print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge

    
