from models.model import *
from GraphCL_data_augmentation import *
from torch_geometric.data import Data
# from torch_geometric.nn import GraphConv


# GraphCL model
# First we will use the basic class of model in model directory of model.py for inserting new algorithm
class GraphCL(Models):
    def __init__(self, **kwargs):
        super(GraphCL, self).__init__(**kwargs)

        self.augment_nodes = False
        self.augment_edges = False
        self.augment_subgraph = False
        self.augment_node_mask = False


    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return True

    @classmethod
    def is_end2end(cls):
        return False

    def check_train_parameters(self):
        space_dtree = {
            'lr': hp.loguniform('lr', -5, 0),
            'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
            'num_epochs': hp.choice('num_epochs', [50, 100, 200]),
            'proj_hidden_dim': hp.choice('proj_hidden_dim', [64, 128, 256]),
            'num_proj_layers': hp.choice('num_proj_layers', [2, 3, 4]),
            'use_l2_norm': hp.choice('use_l2_norm', [True, False]),
            'temperature': hp.loguniform('temperature', -5, 0),
            'num_workers': hp.choice('num_workers', [0, 2, 4, 8])
        }
        return space_dtree

    def train_model(self, **kwargs):
        # Load the adjacency matrix and feature matrix from the dataset
        adj = self.mat_content['Network']
        features = self.mat_content['Attributes']

        # Convert the adjacency matrix and feature matrix to PyTorch Geometric format
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long).t()
        x = torch.tensor(features.toarray(), dtype=torch.float)

        # Apply data augmentation functions according to user preferences
        data = Data(x=x, edge_index=edge_index)
        if self.augment_nodes:
            data = drop_nodes(data)
        if self.augment_edges:
            data = permute_edges(data)
        if self.augment_subgraph:
            data = subgraph(data)
        if self.augment_node_mask:
            data = mask_nodes(data)

        # Create the GraphConv layers
        # conv1 = GraphConv(x.size(1), kwargs['hidden_size'])
        # conv2 = GraphConv(kwargs['hidden_size'], kwargs['hidden_size'])

        # Create the projection MLP
        hidden_sizes = [kwargs['proj_hidden_dim']] * kwargs['num_proj_layers']
        hidden_layers = []
        in_dim = kwargs['hidden_size']
        for hidden_size in hidden_sizes:
            hidden_layers.append(torch.nn.Linear(in_dim, hidden_size))
            hidden_layers.append(torch.nn.BatchNorm1d(hidden_size))
            hidden_layers.append(torch.nn.ReLU(inplace=True))
            in_dim = hidden_size
        projection = torch.nn.Sequential(*hidden_layers, torch.nn.Linear(in_dim, kwargs['projection_size']))

        # Create the contrastive loss criterion
        criterion = torch.nn.CrossEntropyLoss()

        # Move the model and data to the device (GPU or CPU)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.to(device)
        # data = data.to(device)

        # Create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=kwargs['lr'])

        # Train the model
        self.train()
        for epoch in range(kwargs['num_epochs']):
            optimizer.zero_grad()
            z = self.encode(data.x, data.edge_index)
            projection_output = projection(z)
            if self.augment_node_mask:
                loss = self.compute_contrastive_loss(projection_output, data.masked_nodes, criterion)
            else:
                loss = self.compute_contrastive_loss(projection_output, data.node_labels, criterion)
            loss.backward()
            optimizer.step()

            if epoch % kwargs['log_every'] == 0:
                print(f'Epoch {epoch}, loss={loss.item()}')

        # Return the final embedding matrix
        return self.encode(data.x, data.edge_index).detach().cpu().numpy()
