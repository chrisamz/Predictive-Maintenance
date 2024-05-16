import torch
import numpy as np
from torch_geometric.data import Data

# Example feature sizes
node_feature_size = 10  # Example node feature size (e.g., sensor readings)
edge_feature_size = 5   # Example edge feature size (e.g., distances or interaction strengths)

# Generate synthetic node features for industrial components
num_nodes = 50  # Number of components
node_features = torch.randn((num_nodes, node_feature_size))  # Random node features

# Generate synthetic edge indices and features
# Edge indices should be a [2, num_edges] tensor, where each column represents an edge
num_edges = 100  # Number of interactions
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# Generate synthetic edge features
edge_features = torch.randn((num_edges, edge_feature_size))  # Random edge features

# Generate synthetic labels (0 for normal operation, 1 for potential failure)
labels = torch.randint(0, 2, (num_nodes,))  # Binary labels for each component

# Create the graph data object
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=labels)

from torch_geometric.data import DataLoader

# Generate a list of such graphs for training and testing
num_graphs = 1000  # Number of graphs in the dataset
graph_dataset = [graph_data for _ in range(num_graphs)]

# Split into training and test sets
train_dataset = graph_dataset[:800]
test_dataset = graph_dataset[800:]

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PredictiveMaintenanceGNN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_channels, output_channels):
        super(PredictiveMaintenanceGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Perform graph convolution and ReLU.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 2. Aggregate graph features.
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier.
        x = self.lin(x)
        
        return x

# Initialize the model
hidden_channels = 64
output_channels = 1
model = PredictiveMaintenanceGNN(node_feature_size, edge_feature_size, hidden_channels, output_channels)

from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

# Initialize the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.view(-1), data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.view(-1), data.y.float())
            total_loss += loss.item() * data.num_graphs
            pred = (out.view(-1) > 0).float()
            correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Training the model
num_epochs = 50
for epoch in range(1, num_epochs+1):
    train_loss = train()
    test_loss, test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
  
torch.save(model.state_dict(), 'predictive_maintenance_gnn.pth')
