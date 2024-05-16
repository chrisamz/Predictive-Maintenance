pip install torch torch-geometric numpy pandas scikit-learn ortools

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

class RULPredictor(torch.nn.Module):
    def __init__(self, node_feature_size, hidden_dim, output_dim):
        super(RULPredictor, self).__init__()
        self.conv1 = GCNConv(node_feature_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Aggregate graph-level features
        x = self.lin(x)
        return x

# Example usage
node_feature_size = 10
hidden_dim = 64
output_dim = 1  # Predicting remaining useful life
model = RULPredictor(node_feature_size, hidden_dim, output_dim)

import numpy as np

# Generate synthetic data (replace this with actual data loading logic)
def generate_synthetic_rul_data(num_samples=1000):
    data_list = []
    for _ in range(num_samples):
        num_nodes = np.random.randint(5, 20)
        x = torch.rand((num_nodes, node_feature_size))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.tensor([np.random.uniform(0, 100)])  # Remaining useful life
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

# Load the data
data_list = generate_synthetic_rul_data()

# Split the data into training and test sets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

from torch.optim import Adam
from torch.nn import MSELoss

# Initialize the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.view(-1), data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Training the model
num_epochs = 50
for epoch in range(1, num_epochs+1):
    train_loss = train()
    test_loss = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

from ortools.linear_solver import pywraplp

def create_maintenance_schedule(rul_predictions, maintenance_costs, downtime_costs, maintenance_window):
    # Create the linear solver with the CBC backend
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None

    # Variables: 1 if maintenance is performed at time t, 0 otherwise
    maintenance = {}
    for i in range(len(rul_predictions)):
        for t in range(maintenance_window):
            maintenance[(i, t)] = solver.BoolVar(f'maintenance_{i}_{t}')

    # Constraints: each component must be maintained before its predicted RUL
    for i, rul in enumerate(rul_predictions):
        solver.Add(sum(maintenance[(i, t)] * (t <= rul) for t in range(maintenance_window)) >= 1)

    # Objective: minimize the total maintenance and downtime costs
    objective = solver.Objective()
    for i, rul in enumerate(rul_predictions):
        for t in range(maintenance_window):
            objective.SetCoefficient(maintenance[(i, t)], maintenance_costs[i] + downtime_costs[i] * (t > rul))
    objective.SetMinimization()

    # Solve the problem
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        schedule = [(i, t) for i in range(len(rul_predictions)) for t in range(maintenance_window) if maintenance[(i, t)].solution_value() == 1]
        return schedule
    else:
        return None

# Example RUL predictions, maintenance costs, and downtime costs
rul_predictions = [20, 40, 10, 30]  # Predicted remaining useful life for each component
maintenance_costs = [100, 100, 100, 100]  # Maintenance costs for each component
downtime_costs = [500, 500, 500, 500]  # Downtime costs for each component
maintenance_window = 50  # Maintenance scheduling window in days

# Create the maintenance schedule
schedule = create_maintenance_schedule(rul_predictions, maintenance_costs, downtime_costs, maintenance_window)
print(f'Maintenance Schedule: {schedule}')

