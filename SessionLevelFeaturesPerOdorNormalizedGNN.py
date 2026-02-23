#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using session level features (non normalized) from the data to build a classifier 
that classifies single session data.
Z-Scoring done using in training data only.

@author: prashastha
"""


# import packeges
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.utils import shuffle
from scipy.stats import skew
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
# For recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold

# Import PyTorch and PyTorch Geometric
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn


# load data from .mat file
import h5py
data = h5py.File ('data/region_sessions_Notnormalized.mat')
dataAA = h5py.File('data/region_sessions_AA_Notnormalized.mat')
dataNatMixes = h5py.File('data/region_sessions_Nat_Notnormalized.mat')
list(data.keys())


# create 2 array with all sessions from each region
def getRegionalData (data):
    X_pcx = []
    X_plcoa = []
    for sess in data['x_pcx']:
        X_pcx.append(np.array(data[sess[0]]))
    for sess in data['x_plcoa']:
        X_plcoa.append(np.array(data[sess[0]]))
    return X_pcx,X_plcoa




def normalize_session_per_odor_per_neuron(session_data, structure, eps=1e-8):
    """Per-odor, per-neuron z-score within a single session.

    Assumes trials are ordered in contiguous odor blocks:
    [odor1_rep1..odor1_repN | odor2_rep1..odor2_repN | ...].
    For each neuron, each odor block is z-scored independently.
    """
    n_odors, n_reps = structure
    n_neurons = session_data.shape[0]
    normalized_data = np.zeros_like(session_data)

    for neuron_idx in range(n_neurons):
        neuron_data = session_data[neuron_idx, :]
        for odor_idx in range(n_odors):
            start = odor_idx * n_reps
            end = start + n_reps
            odor_block = neuron_data[start:end]
            mean = np.mean(odor_block)
            std = np.std(odor_block)
            normalized_data[neuron_idx, start:end] = (odor_block - mean) / (std + eps)

    return normalized_data
    
# Session structure dict
session_structure = {
    'Mono': (15, 10), # 15 odors, 10 readings per odor. 150 readings per neuron
    'Nat': (13, 10),   # 13 odors, 10 readings per odor. 130 readings per neuron
    'AA': (10, 10)     # 10 odors, 10 readings per odor. 100 readings per neuron
}

def build_normalized_dataset(condition, X_pcx_sessions, X_plcoa_sessions):
    """Build normalized dataset for a given condition."""
    structure = session_structure[condition]
    X_all = []
    y_all = []

    # Process PCx sessions
    for session in X_pcx_sessions:
        print("Processing PCx session shape:", session.shape)
        normalized_session = normalize_session_per_odor_per_neuron(session, structure)
        # Stack (n, 150) to (x * sessions, 150) 2D array
        X_all.append(normalized_session)
        # x * sessions labels
        y_temp = []
        for _ in range(normalized_session.shape[0]):
            y_temp.append('PCx')
        y_all.append(y_temp)

    # Process plCoA sessions
    for session in X_plcoa_sessions:
        print("Processing plCoA session shape:", session.shape)
        normalized_session = normalize_session_per_odor_per_neuron(session, structure)
        X_all.append(normalized_session)
        y_temp = []
        for _ in range(normalized_session.shape[0]):
            y_temp.append('plCoA')
        y_all.append(y_temp)

    return X_all, y_all
    # return np.vstack(X_all), np.array(y_all)


# Build dataset
X_pcx_Mono , X_plcoa_Mono = getRegionalData(data)
X_pcx_Nat , X_plcoa_Nat = getRegionalData(dataNatMixes)
X_pcx_AA , X_plcoa_AA = getRegionalData(dataAA)


# Describe the shape of X_pcx_Mono and X_plcoa_Mono
print("X_pcx_Mono shape:", len(X_pcx_Mono))
print("X_plcoa_Mono shape:", len(X_plcoa_Mono))

# Print shape of first session in X_pcx_Mono
print("First session in X_pcx_Mono shape:", X_pcx_Mono[0].shape)
print("First session in X_plcoa_Mono shape:", X_plcoa_Mono[0].shape)

# Build datasets for all conditions
print("\n" + "="*60)
print("Building Combined Dataset from All Conditions")
print("="*60)

X_mono, y_mono = build_normalized_dataset('Mono', X_pcx_Mono, X_plcoa_Mono)
print(f"Mono dataset: {len(X_mono)} sessions")

X_nat, y_nat = build_normalized_dataset('Nat', X_pcx_Nat, X_plcoa_Nat)
print(f"Nat dataset: {len(X_nat)} sessions")

X_aa, y_aa = build_normalized_dataset('AA', X_pcx_AA, X_plcoa_AA)
print(f"AA dataset: {len(X_aa)} sessions")

# Combine all datasets
X = X_mono + X_nat + X_aa
y_flat_mono = [label[0] if isinstance(label, list) else label for label in y_mono]
y_flat_nat = [label[0] if isinstance(label, list) else label for label in y_nat]
y_flat_aa = [label[0] if isinstance(label, list) else label for label in y_aa]
y = y_flat_mono + y_flat_nat + y_flat_aa
# print("X shape:", X.shape)
# print("y shape:", y.shape)

# Combined dataset information
print(f"\nCombined dataset: {len(X)} total sessions")
print(f"Session trial counts:")
print(f"  Mono sessions (150 trials): {len(X_mono)}")
print(f"  Nat sessions (130 trials): {len(X_nat)}")
print(f"  AA sessions (100 trials): {len(X_aa)}")

# Check class distribution
unique_classes, class_counts = np.unique(y, return_counts=True)
print(f"\nClass distribution:")
for cls, count in zip(unique_classes, class_counts):
    print(f"  {cls}: {count} sessions")

# Show shapes of sessions from different conditions
print(f"\nExample session shapes:")
if len(X_mono) > 0:
    print(f"  Mono session shape: {X_mono[0].shape}")
if len(X_nat) > 0:
    print(f"  Nat session shape: {X_nat[0].shape}")
if len(X_aa) > 0:
    print(f"  AA session shape: {X_aa[0].shape}")

# Feed the data to a GNN model

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_graph_from_session(session_data, correlation_threshold=0.3):
    """
    Convert a session of neural data into a graph representation.
    
    Args:
        session_data: numpy array of shape (n_neurons, n_trials)
                     n_trials can vary: 150 (Mono), 130 (Nat), or 100 (AA)
        correlation_threshold: minimum correlation to create an edge
    
    Returns:
        torch_geometric.data.Data: Graph representation
    """
    n_neurons, n_trials = session_data.shape
    
    # Node features: firing rate statistics per neuron (works for any n_trials)
    node_features = []
    for neuron_idx in range(n_neurons):
        neuron_data = session_data[neuron_idx, :]
        # Features: mean, std, skewness across trials
        mean_fr = np.mean(neuron_data)
        std_fr = np.std(neuron_data)
        skew_fr = skew(neuron_data)
        max_fr = np.max(neuron_data)
        min_fr = np.min(neuron_data)
        node_features.append([mean_fr, std_fr, skew_fr, max_fr, min_fr])
    
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create edges based on correlation between neurons (robust to different n_trials)
    edge_list = []
    correlations = np.corrcoef(session_data)
    
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if abs(correlations[i, j]) > correlation_threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add reverse edge for undirected graph
    
    # If no edges meet threshold, create a few edges to ensure connectivity
    if len(edge_list) == 0:
        # Create edges between neurons with highest correlations
        flat_corr = correlations[np.triu_indices(n_neurons, k=1)]
        top_indices = np.argpartition(np.abs(flat_corr), -min(10, len(flat_corr)))[-min(10, len(flat_corr)):]
        triu_indices = np.triu_indices(n_neurons, k=1)
        for idx in top_indices:
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            edge_list.append([i, j])
            edge_list.append([j, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=node_features, edge_index=edge_index)

def prepare_data_for_gnn(X, y):
    """
    Convert session data into graph format for GNN.
    
    Args:
        X: List of sessions, each session is (n_neurons, n_trials)
        y: List of labels for each session
    
    Returns:
        List of Data objects, encoded labels
    """
    graphs = []
    labels = []
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform([label[0] if isinstance(label, list) else label for label in y])
    
    for session_idx, session_data in enumerate(X):
        graph = create_graph_from_session(session_data)
        graphs.append(graph)
        labels.append(encoded_labels[session_idx])
    
    return graphs, labels, le

class NeuralGNN(torch.nn.Module):
    """
    Graph Neural Network for neural activity classification.
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.1):
        super(NeuralGNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 4, heads=4, concat=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            Dropout(dropout),
            nn.ReLU(),
            Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2) + x1[:, :x2.size(1)]  # Residual connection
        
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        
        # Attention mechanism
        x_att = self.attention(x3, edge_index)
        x_att = F.relu(x_att)
        
        # Global pooling (graph-level representation)
        x_graph = global_mean_pool(x_att, batch)
        
        # Classification
        out = self.classifier(x_graph)
        
        return out

def train_gnn_model(X, y, test_size=0.2, epochs=200, lr=0.001, batch_size=16):
    """
    Train the GNN model and evaluate performance.
    """
    print("Preparing data for GNN...")
    graphs, labels, label_encoder = prepare_data_for_gnn(X, y)
    
    # Split data only if test_size > 0
    if test_size > 0:
        train_graphs, test_graphs, train_labels, test_labels = train_test_split(
            graphs, labels, test_size=test_size, random_state=42, stratify=labels
        )
    else:
        # Use all data for training (external test set provided separately)
        train_graphs, train_labels = graphs, labels
        test_graphs, test_labels = [], []
    
    # Create data loaders
    train_loader = DataLoader(
        [Data(x=g.x, edge_index=g.edge_index, y=torch.tensor(label)) for g, label in zip(train_graphs, train_labels)],
        batch_size=batch_size, shuffle=True
    )
    
    if test_size > 0:
        test_loader = DataLoader(
            [Data(x=g.x, edge_index=g.edge_index, y=torch.tensor(label)) for g, label in zip(test_graphs, test_labels)],
            batch_size=batch_size, shuffle=False
        )
    
    # Initialize model
    input_dim = graphs[0].x.shape[1]  # Number of node features
    model = NeuralGNN(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Model architecture:\n{model}")
    print(f"Training on {len(train_graphs)} graphs" + (f", testing on {len(test_graphs)} graphs" if test_size > 0 else ""))
    
    # Training loop
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')
    
    # Evaluation on internal test set (only if test_size > 0)
    if test_size > 0:
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        test_accuracy = 100 * correct / total
        print(f'\nInternal Test Accuracy: {test_accuracy:.2f}%')
        
        # Convert predictions back to original labels
        predicted_labels = label_encoder.inverse_transform(all_predictions)
        true_labels = label_encoder.inverse_transform(all_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.title(f'GNN Internal Test Confusion Matrix (Accuracy: {test_accuracy:.2f}%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    else:
        test_accuracy = train_accuracies[-1]  # Return final training accuracy if no test set
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()
    
    return model, test_accuracy, label_encoder

# Train/Test Split before training
print("\n" + "="*60)
print("Preparing Train/Test Split")
print("="*60)

# Split the data (y is already flattened)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} sessions")
print(f"Test set: {len(X_test)} sessions")
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)
print(f"Training labels: {dict(zip(train_unique, train_counts))}")
print(f"Test labels: {dict(zip(test_unique, test_counts))}")

# Train the GNN model
print("\n" + "="*60)
print("Training Graph Neural Network")
print("="*60)

model, train_acc, label_encoder = train_gnn_model(X_train, y_train, epochs=1000, test_size=0.0)

# Evaluate on held-out test set
print("\n" + "="*60)
print("Evaluating on Held-out Test Set")
print("="*60)

# Prepare test data
test_graphs, test_labels_encoded, _ = prepare_data_for_gnn(X_test, y_test)
test_loader = DataLoader(
    [Data(x=g.x, edge_index=g.edge_index, y=torch.tensor(label)) for g, label in zip(test_graphs, test_labels_encoded)],
    batch_size=16, shuffle=False
)

# Evaluate model on test set
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)
        total += batch.y.size(0)
        correct += (predicted == batch.y).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Held-out Test Accuracy: {test_accuracy:.2f}%')

# Convert predictions back to original labels
predicted_labels = label_encoder.inverse_transform(all_predictions)
true_labels = label_encoder.inverse_transform(all_labels)

# Plot confusion matrix for held-out test set
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title(f'GNN Held-out Test Confusion Matrix (Accuracy: {test_accuracy:.2f}%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(f"\nFinal GNN Held-out Test Accuracy: {test_accuracy:.2f}%")
