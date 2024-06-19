import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net1(torch.nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
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
    
class Net2(torch.nn.Module):
    def __init__(self, args):
        super(Net2, self).__init__()
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