import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool

class Classify_node(torch.nn.Module):
    def __init__(self, args):
        super(Classify_node, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)
    
class Regress_node(torch.nn.Module):
    def __init__(self, args):
        super(Regress_node, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, 1)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return x
    
class Classify_graph_gc(torch.nn.Module):
    def __init__(self, args):
        super(Classify_graph_gc, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.hidden, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = global_max_pool(x, batch)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)

class Classify_graph_gs(torch.nn.Module):
    def __init__(self, args):
        super(Classify_graph_gs, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.hidden, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, gs, batch):
        X = torch.tensor([]).to(batch.device)
        for g in gs:
            x, edge_index, mask = g.x, g.edge_index, g.mask
            for i in range(self.num_layers):
                x = self.conv[i](x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            X = torch.cat((X, x[mask]), 1)
        x = global_max_pool(X, batch)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)

class Regress_graph(torch.nn.Module):
    def __init__(self, args):
        super(Regress_graph, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.hidden, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, 1)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, gs, batch):
        X = torch.tensor([]).to(batch.device)
        for g in gs:
            x, edge_index, mask = g.x, g.edge_index, g.mask
            for i in range(self.num_layers):
                x = self.conv[i](x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            X = torch.cat((X, x[mask]), 1)
        x = global_max_pool(X, batch)
        x = self.lt1(x)
        return x    
                  

class TransferNet(torch.nn.Module):
    def __init__(self, args, model1):
        super(TransferNet, self).__init__()
        self.num_layers = args.num_layers2
        self.conv = model1.conv
        self.new_lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.new_lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.new_lt1(x)
        return F.log_softmax(x, dim=1)