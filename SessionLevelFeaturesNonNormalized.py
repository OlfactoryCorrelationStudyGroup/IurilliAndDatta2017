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

    for train_idx, test_idx in loo.split(X_all):
        X_train = np.array([X_all[i] for i in train_idx])
        y_train = np.array([y_all[i] for i in train_idx])
        X_test = np.array([X_all[i] for i in test_idx])
        y_test = np.array([y_all[i] for i in test_idx])

        # Fold-wise scaling: fit on training fold, apply to train and test
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        y_preds.append(pred[0])
        y_trues.append(y_test[0])

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

# Build dataset
X_pcx_Mono , X_plcoa_Mono = getRegionalData(data)
X_pcx_Nat , X_plcoa_Nat = getRegionalData(dataNatMixes)
X_pcx_AA , X_plcoa_AA = getRegionalData(dataAA)

# Session structure dict
session_structure = {
    'Mono': (15, 10), # 15 odors, 10 readings per odor. 150 readings per neuron
    'Nat': (13, 10),   # 13 odors, 10 readings per odor. 130 readings per neuron
    'AA': (10, 10)     # 10 odors, 10 readings per odor. 100 readings per neuron
}

def normalize_session_per_odor_neuron(session_data, structure, eps=1e-8):
    """Per-odor, per-neuron z-score within a single session.

    Assumes trials are ordered in contiguous odor blocks:
    [odor1_rep1..odor1_repN | odor2_rep1..odor2_repN | ...].
    For each neuron and odor, z-scores across repeats to remove odor-wise
    baseline/scale within the session while preserving odor structure.
    """
    n_odors, n_reps = structure
    n_neurons, n_trials = session_data.shape
    expected = n_odors * n_reps
    if n_trials < expected:
        raise ValueError(f"Session has {n_trials} trials, expected {expected} for structure {structure}")
    if n_trials > expected:
        session_data = session_data[:, :expected]  # trim extras

    reshaped = session_data.reshape(n_neurons, n_odors, n_reps)
    means = reshaped.mean(axis=2, keepdims=True)
    stds = reshaped.std(axis=2, keepdims=True)
    stds = np.where(stds < eps, eps, stds)
    z = (reshaped - means) / stds
    return z.reshape(n_neurons, expected)


def pad_neuron_dim(session_data, target_neurons, pad_value=0.0):
    """Pad neuron dimension to target_neurons with pad_value."""
    n_neurons, n_trials = session_data.shape
    if n_neurons == target_neurons:
        return session_data
    if n_neurons > target_neurons:
        # If more neurons than target, truncate extras
        return session_data[:target_neurons, :]
    pad_rows = np.full((target_neurons - n_neurons, n_trials), pad_value, dtype=session_data.dtype)
    return np.vstack([session_data, pad_rows])


def build_normalized_dataset(structure_name, X_pcx, X_plcoa, structure_map=session_structure):
    """Build flattened dataset with per-odor per-neuron z-scoring."""
    if structure_name not in structure_map:
        raise ValueError(f"structure_name '{structure_name}' not in {list(structure_map.keys())}")
    structure = structure_map[structure_name]

    # Find maximum neuron count across all sessions to pad/truncate consistently
    max_neurons = 0
    for sess in X_pcx + X_plcoa:
        max_neurons = max(max_neurons, sess.shape[0])

    X_all, y_all = [], []
    for sess in X_pcx:
        norm = normalize_session_per_odor_neuron(sess, structure)
        norm = pad_neuron_dim(norm, max_neurons)
        X_all.append(norm.flatten())
        y_all.append('PCx')
    for sess in X_plcoa:
        norm = normalize_session_per_odor_neuron(sess, structure)
        norm = pad_neuron_dim(norm, max_neurons)
        X_all.append(norm.flatten())
        y_all.append('plCoA')

    return np.array(X_all), np.array(y_all)
'''
### Build normalized dataset for Mono condition
X_all_mono, y_all_mono = build_normalized_dataset('Mono', X_pcx_Mono, X_plcoa_Mono)
print(X_all_mono.shape)
print(y_all_mono.shape)

# Build dataframe from X_all_mono and y_all_mono
df_mono = pd.DataFrame(X_all_mono)
df_mono['Region'] = y_all_mono

# Class sanity check on same class by training and testing on same class
X_same_class = X_all_mono[y_all_mono == 'PCx']
y_same_class = y_all_mono[y_all_mono == 'PCx']
# y_same_class break into two classes artificially
y_same_class = np.array(['PCx_A' if i % 2 == 0 else 'PCx_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")

# Class sanity check on same class by training and testing on same class
X_same_class = X_all_mono[y_all_mono == 'plCoA']
y_same_class = y_all_mono[y_all_mono == 'plCoA']
# y_same_class break into two classes artificially
y_same_class = np.array(['plCoA_A' if i % 2 == 0 else 'plCoA_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")


# Train as svm classifier with leave one out cross validation
y_preds_mono, y_trues_mono, acc_mono = run_leave_one_out_cv(X_all_mono, y_all_mono)
print(f"Mono SVM Leave-One-Out Accuracy: {acc_mono:.4f}")
# Permutation test
real_acc_mono, null_accuracies_mono = permutation_test_loocv(X_all_mono, y_all_mono, n_permutations=1000)
p_value_mono = np.mean([1 if acc >= real_acc_mono else 0 for acc in null_accuracies_mono])
print(f"Mono Permutation Test p-value: {p_value_mono:.4f}")
# Plot confusion matrix
plot_confusion_matrix(y_trues_mono, y_preds_mono, p_value_mono, real_acc_mono)
'''

'''
### Build normalized dataset for Nat condition
X_all_nat, y_all_nat = build_normalized_dataset('Nat', X_pcx_Nat, X_plcoa_Nat)
print(X_all_nat.shape)
print(y_all_nat.shape)

# Build dataframe from X_all_nat and y_all_nat
df_nat = pd.DataFrame(X_all_nat)
df_nat['Region'] = y_all_nat
# Class sanity check on same class by training and testing on same class
X_same_class = X_all_nat[y_all_nat == 'PCx']
y_same_class = y_all_nat[y_all_nat == 'PCx']
# y_same_class break into two classes artificially
y_same_class = np.array(['PCx_A' if i % 2 == 0 else 'PCx_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")

# Class sanity check on same class by training and testing on same class
X_same_class = X_all_nat[y_all_nat == 'plCoA']
y_same_class = y_all_nat[y_all_nat == 'plCoA']
# y_same_class break into two classes artificially
y_same_class = np.array(['plCoA_A' if i % 2 == 0 else 'plCoA_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")


# Train as svm classifier with leave one out cross validation
y_preds_nat, y_trues_nat, acc_nat = run_leave_one_out_cv(X_all_nat, y_all_nat)
print(f"Nat SVM Leave-One-Out Accuracy: {acc_nat:.4f}")
# Permutation test
real_acc_nat, null_accuracies_nat = permutation_test_loocv(X_all_nat, y_all_nat, n_permutations=1000)
p_value_nat = np.mean([1 if acc >= real_acc_nat else 0 for acc in null_accuracies_nat])
print(f"Nat Permutation Test p-value: {p_value_nat:.4f}")
# Plot confusion matrix
plot_confusion_matrix(y_trues_nat, y_preds_nat, p_value_nat, real_acc_nat)
'''


### Build normalized dataset for AA condition
X_all_AA, y_all_AA = build_normalized_dataset('AA', X_pcx_AA, X_plcoa_AA)
print(X_all_AA.shape)
print(y_all_AA.shape)

# Build dataframe from X_all_AA and y_all_AA
df_AA = pd.DataFrame(X_all_AA)
df_AA['Region'] = y_all_AA
# Class sanity check on same class by training and testing on same class
X_same_class = X_all_AA[y_all_AA == 'PCx']
y_same_class = y_all_AA[y_all_AA == 'PCx']
# y_same_class break into two classes artificially
y_same_class = np.array(['PCx_A' if i % 2 == 0 else 'PCx_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")

# Class sanity check on same class by training and testing on same class
X_same_class = X_all_AA[y_all_AA == 'plCoA']
y_same_class = y_all_AA[y_all_AA == 'plCoA']
# y_same_class break into two classes artificially
y_same_class = np.array(['plCoA_A' if i % 2 == 0 else 'plCoA_B' for i in range(len(y_same_class))])
y_preds_same, y_trues_same, acc_same = run_leave_one_out_cv(X_same_class, y_same_class)
print(f"Same Class SVM Leave-One-Out Accuracy: {acc_same:.4f}")


# Train as svm classifier with leave one out cross validation
y_preds_AA, y_trues_AA, acc_AA = run_leave_one_out_cv(X_all_AA, y_all_AA)
print(f"AA SVM Leave-One-Out Accuracy: {acc_AA:.4f}")
# Permutation test
real_acc_AA, null_accuracies_AA = permutation_test_loocv(X_all_AA, y_all_AA, n_permutations=1000)
p_value_AA = np.mean([1 if acc >= real_acc_AA else 0 for acc in null_accuracies_AA])
print(f"AA Permutation Test p-value: {p_value_AA:.4f}")
# Plot confusion matrix
plot_confusion_matrix(y_trues_AA, y_preds_AA, p_value_AA, real_acc_AA)



# print("Shape: PCx sessions:", len(X_pcx_Mono))
# for x in X_pcx_Mono:
#     print("Session shape:", x.shape)
# print("Shape: plCoA sessions:", len(X_plcoa_Mono))
# for x in X_plcoa_Mono:
#     print("Session shape:", x.shape)

# print("Shape: PCx sessions:", len(X_pcx_Nat))
# for x in X_pcx_Nat:
#     print("Session shape:", x.shape)
# print("Shape: plCoA sessions:", len(X_plcoa_Nat))
# for x in X_plcoa_Nat:
#     print("Session shape:", x.shape)

# print("Shape: PCx sessions:", len(X_pcx_AA))
# for x in X_pcx_AA:
#     print("Session shape:", x.shape)
# print("Shape: plCoA sessions:", len(X_plcoa_AA))
# for x in X_plcoa_AA:
#     print("Session shape:", x.shape)

