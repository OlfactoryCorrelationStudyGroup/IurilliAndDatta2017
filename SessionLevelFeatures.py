#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:45:24 2026
Using session level features from the data to build a classifier 
that classifies single session data

@author: ofekh
"""


# import packeges
import numpy as np
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





# load data from .mat file
import h5py
data = h5py.File ('/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_stuff/region_sessions.mat')
dataAA = h5py.File('/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_HelperFunctions/region_sessions_AA.mat')
dataNatMixes = h5py.File('/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_stuff/region_sessions_natMix.mat')
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

# A. Firing Rate Stats - Discriptive features
def mean_response(session_data):
    return np.mean(session_data)

def std_response(session_data):
    return np.std(session_data)

def mean_per_neuron(session_data):
    return np.mean(np.mean(session_data, axis=1))

def std_per_neuron(session_data):
    return np.mean(np.std(session_data, axis=1))

def mean_peak_response(session_data):
    peak_vals = np.max(session_data, axis=1)
    return np.mean(peak_vals)

def fraction_excited(session_data):
    neuron_means = np.mean(session_data, axis=1)
    return np.sum(neuron_means > 0) / len(neuron_means)

def fraction_suppressed(session_data):
    neuron_means = np.mean(session_data, axis=1)
    return np.sum(neuron_means < 0) / len(neuron_means)

def neuron_mean_skew(session_data):
    neuron_means = np.mean(session_data, axis=1)  # shape: (neurons,)
    return skew(neuron_means)

def silent_neuron_fraction(session_data, threshold=0.05):
    neuron_means = np.mean(session_data, axis=1)
    return np.sum(np.abs(neuron_means) < threshold) / len(neuron_means)


# B. Trial Consistency features
def mean_trial_corr(session_data):
    if session_data.shape[1] < 2:
        return np.nan  # not enough trials
    corr = np.corrcoef(session_data.T)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return np.nanmean(corr[mask])

def trial_var_ratio(session_data):
    var_across_trials = np.var(session_data, axis=1)
    total_var = np.var(session_data)
    return np.mean(var_across_trials) / total_var if total_var != 0 else 0

def neuronal_consistency(session_data):
    return np.mean(np.std(session_data, axis=1))

def mean_cosine_similarity(session_data):
    if session_data.shape[0] < 2:
        return np.nan
    cos_sim_matrix = cosine_similarity(session_data)  # shape: (neurons x neurons)
    mask = ~np.eye(cos_sim_matrix.shape[0], dtype=bool)
    return np.mean(cos_sim_matrix[mask])

# C. Corrolation + Dimensionality

def pairwise_neuron_corr_mean(session_data):
    if session_data.shape[0] < 2:
        return np.nan
    corr = np.corrcoef(session_data)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return np.nanmean(corr[mask])

def PC1_explained_var(session_data):
    if min(session_data.shape) < 2:
        return 0
    pca = PCA(n_components=1)
    pca.fit(session_data)
    return pca.explained_variance_ratio_[0]

def dimensionality_ratio(session_data, threshold=0.9):
    n_components = min(session_data.shape)
    pca = PCA(n_components=n_components)
    pca.fit(session_data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_pc = np.searchsorted(cum_var, threshold) + 1
    return n_pc / session_data.shape[0]

def participation_ratio(session_data):
    cov = np.cov(session_data)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]  # avoid div by zero
    return (np.sum(eigvals) ** 2) / np.sum(eigvals**2)

# D. MetaData
def n_neurons(session_data):
    return session_data.shape[0]

def n_trials(session_data):
    return session_data.shape[1]


def extract_all_features(session_data):
    return [
        mean_response(session_data),
        std_response(session_data),
        mean_per_neuron(session_data),
        std_per_neuron(session_data),
        mean_peak_response(session_data),
        fraction_excited(session_data),
        fraction_suppressed(session_data),
        #neuron_mean_skew(session_data),
        #silent_neuron_fraction(session_data),
        mean_trial_corr(session_data),
        trial_var_ratio(session_data),
        neuronal_consistency(session_data),
        #mean_cosine_similarity(session_data),
        pairwise_neuron_corr_mean(session_data),
        PC1_explained_var(session_data),
        dimensionality_ratio(session_data),
        participation_ratio(session_data),
        n_neurons(session_data),
        #n_trials(session_data)
    ]

# SVM with Leave one out testing

def run_leave_one_out_cv(X_all, y_all, clf=None):
    if clf is None:
        clf = SVC(kernel='linear', C=1)

    loo = LeaveOneOut()
    y_preds = []
    y_trues = []

    for train_idx, test_idx in loo.split(X_all):
        X_train = [X_all[i] for i in train_idx]
        y_train = [y_all[i] for i in train_idx]
        X_test = [X_all[i] for i in test_idx]
        y_test = [y_all[i] for i in test_idx]

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

X_pcx , X_plcoa = getRegionalData(data)
X_pcx_Nat , X_plcoa_Nat = getRegionalData(dataNatMixes)
X_pcx_AA , X_plcoa_AA = getRegionalData(dataAA)
X_all = []
y_all = []

for sess in X_plcoa:
    X_all.append(extract_all_features(sess))
    y_all.append('plCoA')
    
for sess in X_plcoa_Nat:
    X_all.append(extract_all_features(sess))
    y_all.append('plCoA') 
    
for sess in X_plcoa_AA:
    X_all.append(extract_all_features(sess))
    y_all.append('plCoA')
    
for sess in X_pcx:
    X_all.append(extract_all_features(sess))
    y_all.append('PCx')
    
for sess in X_pcx_Nat:
    X_all.append(extract_all_features(sess))
    y_all.append('PCx')
    
for sess in X_pcx_AA:
    X_all.append(extract_all_features(sess))
    y_all.append('PCx')

X_all = np.array (X_all)
# X_all → shape: (n_sessions_total, n_features)
# y_all → shape: (n_sessions_total,)

# Scaling all features - Optional
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# PCA before model - If overfitting

X_pca = PCA(n_components=5).fit_transform(X_all)


# Run model

preds, truths, acc = run_leave_one_out_cv(X_all, y_all)
print("LOO Accuracy:", acc)
cm = confusion_matrix(truths, preds, labels=['PCx','plCoA'])
print (cm)



# Run permutation test

real_acc, null_accs = permutation_test_loocv(X_all, y_all, n_permutations=100)
p = sum(a >= real_acc for a in null_accs) / len(null_accs)
print(f"p-value: {p:.4f}")
plot_confusion_matrix(truths, preds, p, real_acc, class_labels=['PCx', 'plCoA'])
# Plot hist

plt.hist(null_accs, bins=20, alpha=0.7, color='gray', label='Null distribution')
plt.axvline(real_acc, color='red', linestyle='--', label='Real accuracy')
plt.xlabel('LOO Accuracy')
plt.ylabel('Frequency')
plt.title('Permutation Test (Session Decoding)')
plt.legend()
plt.show()
