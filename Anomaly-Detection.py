pip install torch torch-geometric numpy pandas scikit-learn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, node_feature_size, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder_conv1 = GCNConv(node_feature_size, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, embedding_dim)
        self.decoder_conv1 = GCNConv(embedding_dim, hidden_dim)
        self.decoder_conv2 = GCNConv(hidden_dim, node_feature_size)

    def encode(self, x, edge_index):
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = self.encoder_conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        z = self.decoder_conv1(z, edge_index)
        z = F.relu(z)
        z = self.decoder_conv2(z, edge_index)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z

# Example usage
node_feature_size = 10
hidden_dim = 32
embedding_dim = 16
model = GraphAutoencoder(node_feature_size, hidden_dim, embedding_dim)

# Assuming `generate_synthetic_data` or actual data loading is defined
data_list = generate_synthetic_data()
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

from torch.optim import Adam

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        x_reconstructed, _ = model(data.x, data.edge_index)
        loss = F.mse_loss(x_reconstructed, data.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            x_reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(x_reconstructed, data.x)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Training the autoencoder
num_epochs = 50
for epoch in range(1, num_epochs+1):
    train_loss = train()
    test_loss = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

def detect_anomalies(data, threshold=0.1):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for graph in data:
            x_reconstructed, _ = model(graph.x, graph.edge_index)
            loss = F.mse_loss(x_reconstructed, graph.x)
            if loss.item() > threshold:
                anomalies.append(graph)
    return anomalies

# Example detection
anomalies = detect_anomalies(test_data, threshold=0.1)
print(f'Number of anomalies detected: {len(anomalies)}')
