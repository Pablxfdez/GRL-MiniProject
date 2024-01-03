import torch
from torch import Tensor
from typing import NamedTuple, Optional, List, Union, Callable
from torch_geometric.data import Data
from torch_geometric.utils import degree

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass

    class LimitsOne(SymmetrySet):
        def __init__(self):
            super().__init__()
            self.hidden_units = 16
            self.num_classes = 2
            self.num_features = 4
            self.num_nodes = 8
            self.graph_class = False

        def makedata(self):
            n_nodes = 16 # There are two connected components, each with 8 nodes
            
            ports = [1,1,2,2] * 8
            colors = [0, 1, 2, 3] * 4

            y = torch.tensor([0]* 8 + [1] * 8)
            edge_index = torch.tensor([[0,1,1,2, 2,3,3,0, 4,5,5,6, 6,7,7,4, 8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,8], [1,0,2,1, 3,2,0,3, 5,4,6,5, 7,6,4,7, 9,8,10,9,11,10,12,11,13,12,14,13,15,14,8,15]], dtype=torch.long)
            x = torch.zeros((n_nodes, 4))
            x[range(n_nodes), colors] = 1
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
            data.ports = torch.tensor(ports).unsqueeze(1)
            return [data]
