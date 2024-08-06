#import os
import tarfile
import pandas as pd
#import protein DB data
from io import StringIO
import pytorch
#import tensorflow 
import torch
import toch.nn.functional as F
from torch_geometric.data import data
from torch_geometric.nn import GCNConv
import torch.optim as optim
import sys
import time
import scipy
import matplotlib.pyplot as plt

# Extract DNA sequencing data from tar file
def extract_dna_data(tar_file_path, extract_path):
    with tarfile.open(tar_file_path, "r:*") as tar:
        tar.extractall(path=extract_path)

#Tar example
tar_file_path = "path/to/your/dna_data.tar"
extract_path = "path/to/extracted_files"
extract_dna_data(tar_file_path, extract_path)

#Import protein data from online database
def import_protein_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        protein_data = pd.read_csv(StringIO(response.text), sep="\t")  # Adjust separator based on the data format
        return protein_data
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

#Example for protein data import
protein_data_url = "http://example.com/protein_data.tsv"
protein_data = import_protein_data(protein_data_url)

#Load and preprocess dna data (must be .csv format)
def load_dna_data(extract_path):
    dna_data_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]
    dna_data_list = [pd.read_csv(os.path.join(extract_path, f)) for f in dna_data_files]
    
#Combining dna/protein data into a single dataset
    dna_data = pd.concat(dna_data_list, ignore_index=True)
    return dna_data

#Example 
dna_data = load_dna_data(extract_path)

def matrix_to_graph(matrix):

    matrix = torch.tensor(matrix, dtype=torch.float32)
    edge_index = (matrix > 0).nonzero(as_tuple=False).t().contiguous()
    
    edge_attr = matrix[edge_index[0], edge_index[1]]
    
    num_nodes = matrix.size(0)
    node_features = torch.eye(num_nodes)  
#Create a Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
#1st conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
#2nd conv
        x = self.conv2(x, edge_index)
        
        return x

#Example
input_dim = graph_data.num_features
hidden_dim = 16
output_dim = 2  

model = GNN(input_dim, hidden_dim, output_dim)

# Forward pass with the graph data
output = model(graph_data)
print(output)

# Example training loop (ensure nodes are labelled)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    out = model(graph_data)
    
    
    labels = torch.tensor([0, 1, 0, 1, 0])  
    loss = criterion(out, labels)
    
#Optimisation
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
