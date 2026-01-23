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
def run_leave_one_out_cv(X_all, y_all, clf=None, verbose=False):
    if clf is None:
        clf = SVC(kernel='rbf', C=1)

    loo = LeaveOneOut()
    y_preds = []
    y_trues = []
    
    n_samples = len(X_all)
    for i, (train_idx, test_idx) in enumerate(loo.split(X_all)):
        # Use numpy indexing directly - much faster than list comprehension
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        # Fold-wise scaling: fit on training fold, apply to train and test
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        y_preds.append(pred[0])
        y_trues.append(y_test[0])
        
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Completed {i+1}/{n_samples} folds")

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
def permutation_test_loocv(X_all, y_all, n_permutations=100, clf=None, use_kfold=False, n_folds=5, verbose=True):
    """Run permutation test with LOOCV or k-fold CV.
    
    Args:
        X_all: Feature matrix
        y_all: Labels
        n_permutations: Number of permutations to run
        clf: Classifier (default: linear SVM)
        use_kfold: If True, use k-fold CV instead of LOOCV for permutations (much faster)
        n_folds: Number of folds for k-fold CV
        verbose: Print progress
    """
    if verbose:
        print(f"Running real accuracy with LOOCV on {len(X_all)} samples...")
    real_preds, real_trues, real_acc = run_leave_one_out_cv(X_all, y_all, clf=clf)
    
    if verbose:
        print(f"Real accuracy: {real_acc:.4f}")
        print(f"Running {n_permutations} permutations with {'k-fold CV' if use_kfold else 'LOOCV'}...")
    
    null_accuracies = []
    for perm_idx in range(n_permutations):
        if verbose:
            print(f"Permutation {perm_idx+1}/{n_permutations}...")
        y_shuffled = shuffle(y_all, random_state=None)
        
        if use_kfold:
            # Use k-fold for permutations (much faster)
            acc = run_kfold_cv(X_all, y_shuffled, clf=clf, n_folds=n_folds, verbose=verbose)
        else:
            _, _, acc = run_leave_one_out_cv(X_all, y_shuffled, clf=clf)
        
        null_accuracies.append(acc)
        if verbose:
            print(f"  Accuracy: {acc:.4f}")

    return real_acc, null_accuracies

# Faster k-fold CV alternative for permutation tests
def run_kfold_cv(X_all, y_all, clf=None, n_folds=5, verbose=False):
    """Run k-fold CV (much faster than LOOCV for permutation tests)."""
    if clf is None:
        clf = SVC(kernel='rbf', C=1)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_preds = []
    y_trues = []
    
    for train_idx, test_idx in skf.split(X_all, y_all):
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        y_preds.extend(preds)
        y_trues.extend(y_test)
        if verbose:
            print(f"  Completed fold with {len(test_idx)} samples")
    acc = accuracy_score(y_trues, y_preds)
    return acc

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


# X, y = build_normalized_dataset('Mono', X_pcx_Mono, X_plcoa_Mono)
# print("X shape:", X.shape)
# print("y shape:", y.shape)

# X, y = build_normalized_dataset('AA', X_pcx_AA, X_plcoa_AA)
# print("X shape:", X.shape)
# print("y shape:", y.shape)

X, y = build_normalized_dataset('Nat', X_pcx_Nat, X_plcoa_Nat)
print("X shape:", X.shape)
print("y shape:", y.shape)

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
# Use use_kfold=True for much faster permutation tests (5-10x speedup)
real_acc, null_accuracies = permutation_test_loocv(X, y, n_permutations=100, use_kfold=True, n_folds=5)
p_value = np.mean([1 if acc >= real_acc else 0 for acc in null_accuracies])
print(f"\nReal accuracy: {real_acc:.4f}, p-value: {p_value:.4f}")

# Plot histogram of null accuracies
plt.figure(figsize=(8, 5))
plt.hist(null_accuracies, bins=20, color='skyblue', edgecolor='black')
plt.axvline(real_acc, color='red', linestyle='dashed', linewidth=2, label=f'Real Accuracy: {real_acc:.4f}')
plt.title('Null Distribution of Accuracies from Permutation Test')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.show()
