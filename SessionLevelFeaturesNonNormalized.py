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


# SVM with Leave one out testing
def run_leave_one_out_cv(X_all, y_all, clf=None):
    if clf is None:
        clf = SVC(kernel='linear', C=1)

    loo = LeaveOneOut()
    y_preds = []
    y_trues = []
    skipped_folds = 0

    for train_idx, test_idx in loo.split(X_all):
        X_train = X_all[train_idx]
        y_train = np.array(y_all)[train_idx]
        X_test = X_all[test_idx]
        y_test = np.array(y_all)[test_idx]

        # Check if training set has at least one sample from each class
        unique_train_classes = np.unique(y_train)
        unique_all_classes = np.unique(y_all)
        
        if len(unique_train_classes) < len(unique_all_classes):
            # Skip this fold - training set doesn't have all classes
            skipped_folds += 1
            # For skipped folds, predict randomly from available classes
            y_preds.append(np.random.choice(unique_all_classes))
            y_trues.append(y_test[0])
            continue
        
        # Additional check: ensure we have at least 2 samples of each class for stable training
        train_class_counts = {cls: np.sum(y_train == cls) for cls in unique_train_classes}
        min_count = min(train_class_counts.values())
        
        if min_count == 1:
            # Skip this fold - one class has only 1 sample, which may cause issues
            skipped_folds += 1
            # For skipped folds, predict the most common class in the training set
            most_common_class = max(train_class_counts, key=train_class_counts.get)
            y_preds.append(most_common_class)
            y_trues.append(y_test[0])
            continue

        # Fold-wise scaling: fit on training fold, apply to train and test
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        y_preds.append(pred[0])
        y_trues.append(y_test[0])

    if skipped_folds > 0:
        print(f"Warning: Skipped {skipped_folds} folds due to missing classes in training set")

    acc = accuracy_score(y_trues, y_preds)
    return y_preds, y_trues, acc

def plot_confusion_matrix(y_true, y_pred, p, real_acc, class_labels=['PCx', 'plCoA'], normalize=True,):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / cm_sum if normalize else cm

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, cbar=False, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f"Confusion Matrix,acc: {np.round(real_acc,4)} p-value: {p:.4f}", fontsize=14)
    plt.yticks(rotation=0)
    plt.show()

# Permutation test function
def permutation_test_loocv(X_all, y_all, n_permutations=100, clf=None):

    real_preds, real_trues, real_acc = run_leave_one_out_cv(X_all, y_all, clf=clf)
    null_accuracies = []

    for _ in range(n_permutations):
        y_shuffled = shuffle(y_all, random_state=None)
        _, _, acc = run_leave_one_out_cv(X_all, y_shuffled, clf=clf)
        null_accuracies.append(acc)

    return real_acc, null_accuracies

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
        for _ in range(normalized_session.shape[0]):
            y_all.append('PCx')

    # Process plCoA sessions
    for session in X_plcoa_sessions:
        print("Processing plCoA session shape:", session.shape)
        normalized_session = normalize_session_per_odor_per_neuron(session, structure)
        X_all.append(normalized_session)
        for _ in range(normalized_session.shape[0]):
            y_all.append('plCoA')

    return np.vstack(X_all), np.array(y_all)


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

X, y = build_normalized_dataset('Mono', X_pcx_Mono, X_plcoa_Mono)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Number of data points
print("Number of data points:", X.shape[0])
print("Number of features:", X.shape[1])


# Sanity check for each class
print("Sanity check - class distribution:")
# Check class distribution
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
for class_label, count in class_distribution.items():
    print(f"Class {class_label}: {count} samples")

# Split data in same class to 2 different hypothetical classes and check LOOCV accuracy: PCx
X_pcx_only = X[y == 'PCx']
y_pcx_only = y[y == 'PCx'].copy().astype('U10')  # Use Unicode string with max length 10
print(f"y_pcx_only dtype: {y_pcx_only.dtype}, shape: {y_pcx_only.shape}")
# Random shuffle indices
indices_pcx = list(range(len(X_pcx_only)))
random.shuffle(indices_pcx)
y_pcx_only[indices_pcx[:len(y_pcx_only)//2]] = 'PCx_A'
y_pcx_only[indices_pcx[len(y_pcx_only)//2:]] = 'PCx_B'
# Train model and check accuracy
_, _, acc = run_leave_one_out_cv(X_pcx_only, y_pcx_only)
print(f"LOOCV accuracy after splitting PCx into 2 classes: {acc:.4f}")


# Split data in same class to 2 different hypothetical classes and check LOOCV accuracy: plCoA
X_plcoa_only = X[y == 'plCoA']
y_plcoa_only = y[y == 'plCoA'].copy().astype('U10')  # Use Unicode string with max length 10
print(f"y_plcoa_only dtype: {y_plcoa_only.dtype}, shape: {y_plcoa_only.shape}")
# Random shuffle indices
indices_plcoa = list(range(len(X_plcoa_only)))
random.shuffle(indices_plcoa)
y_plcoa_only[indices_plcoa[:len(y_plcoa_only)//2]] = 'plCoA_A'
y_plcoa_only[indices_plcoa[len(y_plcoa_only)//2:]] = 'plCoA_B'
# Train model and check accuracy
_, _, acc = run_leave_one_out_cv(X_plcoa_only, y_plcoa_only)
print(f"LOOCV accuracy after splitting plCoA into 2 classes: {acc:.4f}")




# Train SVM with LOOCV and plot confusion matrix
y_pred, y_true, real_acc = run_leave_one_out_cv(X, y)
print(f"Leave-One-Out CV Accuracy: {real_acc:.4f}")
plot_confusion_matrix(y_true, y_pred, p=0.0, real_acc=real_acc)

# Run permutation test with LOOCV
real_acc, null_accuracies = permutation_test_loocv(X, y, n_permutations=100)
p_value = np.mean([1 if acc >= real_acc else 0 for acc in null_accuracies])
print(f"Real accuracy: {real_acc:.4f}, p-value: {p_value:.4f}")
