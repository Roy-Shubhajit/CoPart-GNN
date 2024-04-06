import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net1(torch.nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        z = self.lt1(x)
        return F.log_softmax(z, dim=1), x
    
class Net2(torch.nn.Module):
    def __init__(self, args):
        super(Net2, self).__init__()
        self.w0 = torch.nn.Parameter(torch.Tensor(args.num_features, 1))
        self.b0 = torch.nn.Parameter(torch.Tensor(args.num_features, args.num_features))
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w0)
        torch.nn.init.xavier_uniform_(self.b0)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, E_meta):
        new_w0 = torch.matmul(self.w0, E_meta)
        new_w0 = torch.add(new_w0, self.b0)
        new_w0 = F.relu(new_w0)
        x = torch.matmul(x, new_w0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)