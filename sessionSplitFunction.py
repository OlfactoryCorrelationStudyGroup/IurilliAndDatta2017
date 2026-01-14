#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 12:50:31 2026

@author: ofekh
"""

# import packeges
import numpy as np
from sklearn.svm import SVC

# load data from .mat file
import h5py
data = h5py.File ('/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_stuff/region_sessions.mat')

list(data.keys())

# # Get references to PCX data arrays from the data dictionary
# X_pcx_refs = data['x_pcx']
# X_pcx = []

# # Loop through each reference in the first element of x_pcx_refs
# # and append the corresponding numpy array from the data dictionary to X_pcx
# for ref in X_pcx_refs[0]:
#     X_pcx.append(np.array(data[ref], dtype = np.float32))

# # Get references to PLCOA data arrays from the data dictionary
# X_plcoa_refs = data['x_plcoa']
# X_plcoa = []

# # Loop through each reference in the first element of x_plcoa_refs
# # and append the corresponding numpy array from the data dictionary to X_plcoa
# for ref in X_plcoa_refs[0]:
#     X_plcoa.append(np.array(data[ref] , dtype = np.float32))
    

# create 2 array with all sessions from each region
def getRegionalData (data):
    X_pcx = []
    X_plcoa = []
    for sess in data['x_pcx']:
        X_pcx.append(np.array(data[sess[0]]))
    for sess in data['x_plcoa']:
        X_plcoa.append(np.array(data[sess[0]]))
    return X_pcx,X_plcoa


def build_dataset_with_holdout (X_pcx, X_plcoa, N , sess_index = 1 , rng = None):
    # labels : (0 = pcx , 1 = plcoa)
    if rng is None:
        rng = np.random.default_rng()
    X , y = [],[]
    X_held,y_held = [],[]
    

    # Exclude the k session
    if sess_index is not None:
        
        X_pcx_test_session = X_pcx[sess_index]
        X_pcx_train = X_pcx[:sess_index] + X_pcx[sess_index+1:]
        
        X_plcoa_test_session = X_plcoa[sess_index]
        X_plcoa_train = X_plcoa[:sess_index] + X_plcoa[sess_index+1:]
        # Building separate session for testing true population
        X_pcx_test_session = X_pcx_test_session.T
        X_plcoa_test_session = X_plcoa_test_session.T
        
        N = min (X_pcx_test_session.shape[1],X_plcoa_test_session.shape[1], N)
        
    else:
        X_pcx_train = X_pcx
        X_plcoa_train = X_plcoa
    
    for sess in X_pcx_train:
        sess = sess.T  # Trasposing into expected trials X Neurons format
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:, idx])
        y.append(np.zeros(sess.shape[0]))

    for sess in X_plcoa_train:
        sess = sess.T  # Trasposing into expected trials X Neurons format
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:, idx])
        y.append(np.ones(sess.shape[0]))
        
        
    if sess_index is not None:
        idx_pcx = rng.choice(X_pcx_test_session.shape[1], N, replace=False) 
        idx_plcoa = rng.choice(X_plcoa_test_session.shape[1], N, replace=False)
        X_held = np.vstack([
        X_pcx_test_session[:, idx_pcx],
        X_plcoa_test_session[:, idx_plcoa]])
    
        y_held = np.concatenate([
        np.zeros(X_pcx_test_session.shape[0]),
        np.ones(X_plcoa_test_session.shape[0])])
        return np.vstack(X), np.concatenate(y) , X_held, y_held
    return np.vstack(X), np.concatenate(y) , None , None


#X , y, X_held, y_held = build_dataset_with_holdout(X_pcx,X_plcoa, 10)
#X.shape , y.shape


# rbf SVM classification

def singleSVMClassification (X,y,X_held = None ,y_held = None, test_size = 0.1, random_state = 42):

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    svc_orig = SVC(kernel='rbf')
    svc_orig.fit(X_train, y_train)
    accuracy_orig = svc_orig.score(X_test,y_test)
    if X_held is not None or y_held is not None:
        accuracy_held = svc_orig.score(X_held,y_held)
        print(f'svc accuracy on true population: {accuracy_held * 100:.2f}%')
    else:
        accuracy_held = None
    print(f'svc accuracy on pseudo population: {accuracy_orig * 100:.2f}%')
    
    return accuracy_orig , accuracy_held


"loop through the session, always leaving a differnt session for testing"

# 1 - Run build dataset builder for for each iteration with sess_index set to the loop index
# 2 - Run the classification function keeping the accuracies in a variable: "pseudoAcc" "trueAcc"
# 3 - Plot the accuracies as a distribution

X_pcx , X_plcoa = getRegionalData(data)
import matplotlib.pyplot as plt
n_sessions = min(len(X_pcx), len(X_plcoa))
pseudoAcc, trueAcc = [],[]
for i in range(n_sessions):
    
    out  = build_dataset_with_holdout(X_pcx,X_plcoa, 10, sess_index = i) # Skip iteration if invalid
    if out is None:
        continue
    
    X , y, X_held, y_held = out
    # Run SVM classification
    
    p_Acc , t_Acc = singleSVMClassification(X, y, X_held, y_held)
    pseudoAcc.append(p_Acc)
    trueAcc.append(t_Acc)
    

"plots plots plots"    

# Distribution

plt.figure(figsize=(7,5))

plt.hist(pseudoAcc, bins=10, alpha=0.7, label='Pseudo-population')
plt.hist(trueAcc, bins=10, alpha=0.7, label='Held-out session')

plt.axvline(0.5, color='k', linestyle='--', label='Chance (50%)')

plt.xlabel('Classification accuracy')
plt.ylabel('Count')
plt.title('Accuracy distributions across held-out sessions')
plt.legend()
plt.tight_layout()
plt.show()    

# Boxplot

plt.figure(figsize=(6,5))

bp = plt.boxplot(
    [pseudoAcc, trueAcc],
    labels=['Pseudo-population', 'Held-out session testing'],
    patch_artist=True,  # Still needed if you want colored boxes
    showfliers=True
)

# Change the color of the median lines
median_colors = ['darkblue', 'darkgreen']
for median, color in zip(bp['medians'], median_colors):
    median.set_color(color)
    median.set_linewidth(2)  # optional, for better visibility

plt.axhline(0.5, color='k', linestyle='--')

plt.ylabel('Classification accuracy')
plt.title('Decoding performance across sessions')
plt.tight_layout()
plt.show()

"Resample using variable N with all population"

# Optimising N for best performance
# Loop though building traing and testing to find the best N value : [5,10,15]

N_values = [5,10,15,17,20,22,25]
accuracies_N_varies = []
for i in range(len(N_values)):
    
    out  = build_dataset_with_holdout\
        (X_pcx,X_plcoa, N_values[i], sess_index = None)
        
    if out is None:
        continue
    X , y , X_held, y_held = out
    
    temp = singleSVMClassification(X, y)
    accuracies_N_varies.append(temp)
    
pseudo_acc = np.array([a[0] for a in accuracies_N_varies]) * 100

plt.figure(figsize=(6,4))

plt.plot(N_values, pseudo_acc, marker='o', linewidth=2, label='Pseudo-population')

plt.axhline(50, color='k', linestyle='--', linewidth=1, label='Chance')

plt.xlabel('Number of neurons (N)')
plt.ylabel('Classification accuracy %')
plt.title('Decoding accuracy vs subpopulation size')
plt.ylim(40, 100)
plt.legend()
plt.tight_layout()
plt.show()

"Using the optimal N to resample 50 times to get a full distribution with all data"
accuracies_dist = []
rng = np.random.default_rng()
for i in range(50):
    out  = build_dataset_with_holdout\
        (X_pcx,X_plcoa, 20 , sess_index = None,rng= rng)
    if out is None:
        continue
    X , y , X_held, y_held = out
    
    temp = singleSVMClassification(X, y)
    accuracies_dist.append(temp)
    
    
"Plotting"

acc = np.array([i[0] for i in accuracies_dist])
# histogram

plt.figure(figsize=(6,4))
plt.hist(acc, bins=10, color='steelblue', edgecolor='black')
plt.axvline(0.5, color='k', linestyle='--')

plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Accuracy distribution across resamples')
plt.tight_layout()
plt.show()


"Permutation test - building Null distribution"
observed_mean = np.mean(acc)


perm_means = []

rng = np.random.default_rng(123)

for p in range(50):
    perm_acc = []
    
    for i in range (50):
        out  = build_dataset_with_holdout\
            (X_pcx,X_plcoa, 20 , sess_index = None,rng= rng)
        if out is None:
            continue
        X , y , X_held, y_held = out
        
        y_perm = rng.permutation(y)
        
        P_acc = singleSVMClassification(X, y_perm)
        t = P_acc[0]
        
        perm_acc.append(t)
        
    perm_means.append(np.mean(perm_acc))
    


# Compute p-value

perm_means = np.array(perm_means)

p_value = (1 + np.sum(perm_means >= observed_mean)) / (1 + len(perm_means))
    
print ("Permutation p-value:  ", p_value)

# Plotting

plt.figure(figsize=(6,4))
plt.hist(perm_means, bins=20, alpha=0.7, label='Null (permuted)')
plt.axvline(observed_mean, color='r', linewidth=2, label='Observed')

plt.xlabel('Mean decoding accuracy')
plt.ylabel('Count')
plt.title(f'Permutation test, p-value:  {round(p_value,3)}')
plt.legend()
plt.tight_layout()
plt.show()
