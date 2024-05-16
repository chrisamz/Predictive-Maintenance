pip install torch torch-geometric numpy pandas scikit-learn

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class PredictiveMaintenanceGNN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_channels, output_channels):
        super(PredictiveMaintenanceGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, edge_index, batch):
        # 1. Perform graph convolution and ReLU.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 2. Aggregate graph features.
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier.
        x = self.lin(x)
        
        return x

# Define the model parameters
node_feature_size = 10  # Example feature size for nodes
edge_feature_size = 5   # Example feature size for edges
hidden_channels = 64    # Hidden channels for the GNN
output_channels = 1     # Output channels (binary classification)

model = PredictiveMaintenanceGNN(node_feature_size, edge_feature_size, hidden_channels, output_channels)

import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data (replace this with your actual data loading logic)
def generate_synthetic_data(num_samples=1000):
    data_list = []
    for _ in range(num_samples):
        num_nodes = np.random.randint(5, 20)
        x = torch.rand((num_nodes, node_feature_size))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.tensor([np.random.randint(0, 2)])  # Binary label
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

# Load the data
data_list = generate_synthetic_data()

# Split the data into training and test sets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

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
        out = model(data.x, data.edge_index, data.batch)
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
            out = model(data.x, data.edge_index, data.batch)
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

model = PredictiveMaintenanceGNN(node_feature_size, edge_feature_size, hidden_channels, output_channels)
model.load_state_dict(torch.load('predictive_maintenance_gnn.pth'))
