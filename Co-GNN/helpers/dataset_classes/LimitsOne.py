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
        self.data = self.makedata()  # Initialize the data attribute during object creation

    def makedata(self):
        # [Existing implementation of makedata]
        
        # Create the data object
        data = self.create_data_object()
        # Modify the data object as needed (e.g., add ports and features)
        data = self.addports(data)
        data = self.makefeatures(data)
        return data

    def create_data_object(self):
 
        n_nodes = 16
        ports = [1,1,2,2] * 8
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0]* 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2, 2,3,3,0, 4,5,5,6, 6,7,7,4, 8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,8], [1,0,2,1, 3,2,0,3, 5,4,6,5, 7,6,4,7, 9,8,10,9,11,10,12,11,13,12,14,13,15,14,8,15]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False
        self.data = self.makedata()  # Initialize the data attribute during object creation

    def makedata(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        ports = ([1,1,2,2,1,1,2,2] * 2 + [3,3,3,3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 1,3,5,7, 8,9,9,10,10,11,11,8, 12,13,13,14,14,15,15,12, 9,15,11,13], [1,0,2,1,3,2,0,3, 5,4,6,5,7,6,4,7, 3,1,7,5, 9,8,10,9,11,10,8,11, 13,12,14,13,15,14,12,15, 15,9,13,11]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        return data

class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False
        self.data = self.makedata()  # Initialize the data attribute during object creation

    def makedata(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0]==n]:
                    for nb2 in data.edge_index[1][data.edge_index[0]==n]:
                        if torch.logical_and(data.edge_index[0]==nb1, data.edge_index[1]==nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = torch.tensor(labels)

        data = self.addports(data)
        data = self.makefeatures(data)
        return data