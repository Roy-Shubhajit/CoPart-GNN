import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net1(torch.nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        self.num_layers = args.num_layers1
        self.modules = torch.nn.ModuleList()
        self.modules.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.modules.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.modules:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.modules[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        z = self.lt1(x)
        return z, x
    
class Net2(torch.nn.Module):
    def __init__(self, args):
        super(Net2, self).__init__()
        self.num_layers = args.num_layers2
        self.w0 = torch.nn.Parameter(torch.Tensor(args.num_features, 1))
        self.b0 = torch.nn.Parameter(torch.Tensor(args.num_features, args.num_features))
        self.modules = torch.nn.ModuleList()
        self.modules.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.modules.append(GCNConv(args.hidden, args.hidden))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w0)
        torch.nn.init.xavier_uniform_(self.b0)
        for module in self.modules:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index, E_meta):
        new_w0 = torch.matmul(self.w0, E_meta)
        new_w0 = torch.add(new_w0, self.b0)
        new_w0 = F.relu(new_w0)
        x = torch.matmul(x, new_w0)
        for i in range(self.num_layers):
            x = self.modules[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)