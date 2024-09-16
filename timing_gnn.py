import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as Loader
from classes.lut import LUT
from classes.net import Net
from classes.node import Node
from classes.pin import Pin
from classes.cell import Cell
import matplotlib.pyplot as plt


# Read the CSV file
df = pd.read_csv('timing_data_Netlists_0912.csv')

with open('netlist_list_dataset_10k.pkl', 'rb') as file:
    netlist_list_dataset = pickle.load(file)

for i in range(len(netlist_list_dataset)):
    instances = df[df['Netlist']==i]['Inst']
    graph = netlist_list_dataset["gen_netlist_"+str(i)]
    name_to_node_map = {node.name: node for node in graph}
    
    for n in name_to_node_map:
        if name_to_node_map[n].is_input:
            name_to_node_map[n].cell_delay = 0
            name_to_node_map[n].input_slew = 0.002
            name_to_node_map[n].load_capacitance = 0
            name_to_node_map[n].fanin = 0
            name_to_node_map[n].fanout = len(graph[name_to_node_map[n]])
            name_to_node_map[n].output_slew = 0.002
        else:
            netlist = df[df['Netlist']==i]
            inst_row = netlist[netlist['Inst'] == n]
            name_to_node_map[n].cell_delay = inst_row['Cell_delay'].item()
            name_to_node_map[n].input_slew = inst_row['In_slew'].item()
            name_to_node_map[n].load_capacitance = inst_row['Load_cap'].item()
            name_to_node_map[n].fanin = inst_row['Fanin'].item()
            name_to_node_map[n].fanout = inst_row['Fanout'].item()
            name_to_node_map[n].output_slew = inst_row['Out_slew'].item()      

encoded_cell_names = {}
for i in range(1,len(comb_cells_for_output)+1):
    b_num = format(i,"09b")
    encoded_cell_names[comb_cells_for_output[i-1].name] = np.array([float(b) for b in b_num])



# Initialize the scaler
scaler = MinMaxScaler()

feature_array = []
for g_name, graph in netlist_list_dataset.items():
    for node in graph:
        if node.is_input:
            encoded_cell_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            encoded_cell_array = encoded_cell_names[node.cell.name]
            
        features = np.array([node.num_inputs, node.num_outputs, node.is_input, node.is_output, node.input_slew, node.load_capacitance, 
                             node.fanin, node.fanout, node.cell_delay, node.output_slew])
        features = np.concatenate((encoded_cell_array,features))
        feature_array.append(features)

scaled_data = scaler.fit_transform(feature_array)

curent_index = 0
next_index = 0

dataset = []

def get_edge_index(graph):
    # Create a mapping from node objects to indices
    node_to_index = {node: idx for idx, node in enumerate(graph.keys())}
    
    # Lists to hold the source and target indices
    source_indices = []
    target_indices = []
    
    # Populate the source and target lists
    for source_node, target_nodes in graph.items():
        source_idx = node_to_index[source_node]
        for target_node in target_nodes:
            target_idx = node_to_index[target_node]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
    
    # Convert lists to tensors
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    
    return edge_index

for g_name, graph in netlist_list_dataset.items():
    num_nodes = len(graph)
    x= torch.from_numpy(scaled_data[next_index : next_index + num_nodes,:-2]).to(torch.float32)
    y= torch.from_numpy(scaled_data[next_index : next_index + num_nodes,-2:]).to(torch.float32)
    next_index = next_index + num_nodes
    edge_index = get_edge_index(graph)
    dataset.append(Data(x=x,y=y,edge_index=edge_index))

max_size=0
for y in dataset:
    max_size = max(max_size,y.x.size()[0])

def pad_dataset(dataset,max_size):
    for data in dataset:
        num_nodes = data.x.size(0)
        num_feats = data.x.size(1)
        num_feats_y = data.y.size(1)
        num_rows_needed = max_size-num_nodes
        
        zeros_x = torch.zeros((num_rows_needed,num_feats),dtype=torch.float32)
        data.x = torch.cat((data.x,zeros_x),dim=0)
        
        zeros_y = torch.zeros((num_rows_needed,num_feats_y),dtype=torch.float32)
        data.y = torch.cat((data.y,zeros_y),dim=0)
        
pad_dataset(dataset,max_size)

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(input_dim, 128)  
        self.conv2 = GraphConv(128, 256)
        self.conv3 = GraphConv(256, 128)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Move tensors to the correct device
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)

        # Apply GraphConv layers normally
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        x = F.relu(self.fc(x))  # Applying activation function

        # Reshape x to [num_graphs, output_dim]
        x = x.view(-1, output_dim)

        return x

# Ensure model is on CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_list=[]
# Define model and move to device
input_dim = 17
hidden_dim = 250
output_dim = 31*2
model = GNNModel(input_dim, hidden_dim, output_dim).to(device)

# Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define dataset and loaders
train_ratio = 0.8
num_train = int(len(dataset) * train_ratio)
num_test = len(dataset) - num_train

# Using random_split to create training and testing datasets
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

# Create DataLoader instances for training and testing
train_loader = Loader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
test_loader = Loader(test_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=2)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)

        # Flatten the outputs for each data point in the batch
        outputs_flat = outputs.view(-1)

        # Flatten and reshape the labels for each data point in the batch
        labels = batch.y.float().view(-1).to(device)  # Cast to torch.float and move to device

        loss = criterion(outputs_flat, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    average_loss = total_loss / len(train_loader)
    loss_list.append(average_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {average_loss:.4f}')

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(loss_list, linestyle='-', color='b', label='Loss')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.ylim(0.0003, 0.0023)
# Show the plot
plt.show()


# Test loop
test_loader = Loader(test_dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=2)
model.eval()
with torch.no_grad():
    total_test_loss = 0.0
    predictions = []
    true_labels = []

    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        test_outputs = model(test_batch)

        # Flatten the outputs for each data point in the batch
        test_outputs_flat = test_outputs.view(-1)

        # Flatten and reshape the labels for each data point in the batch
        test_labels = test_batch.y.float().view(-1)

        test_loss = criterion(test_outputs_flat, test_labels)
        total_test_loss += test_loss.item()

        # Save predictions and true labels for further analysis if needed
        predictions.extend(test_outputs_flat.cpu().numpy())
        true_labels.extend(test_labels.cpu().numpy())
        
        print(f'Test Loss: {test_loss.item():.4f}')

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss:.4f}')