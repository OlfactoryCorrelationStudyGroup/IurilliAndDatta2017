#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 12:50:31 2026

@author: ofekh
"""

# import packeges
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from numpy.random import default_rng
import matplotlib.pyplot as plt
import h5py

plt.rcParams['figure.dpi'] = 250            # Set default resolution



# load data from .mat file
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


def build_dataset_with_holdout(X_pcx, X_plcoa, N, sess_index= None, rng=None):
    """
    labels: 0 = PCx, 1 = plCoA
    """

    if rng is None:
        rng = np.random.default_rng()

    X, y = [], []
    X_held, y_held = [], []

    # Exclude the held-out session
    if sess_index is not None:
        X_pcx_test_session = X_pcx[sess_index]
        X_plcoa_test_session = X_plcoa[sess_index]

        X_pcx_train = X_pcx[:sess_index] + X_pcx[sess_index + 1:]
        X_plcoa_train = X_plcoa[:sess_index] + X_plcoa[sess_index + 1:]

        # Transpose test sessions to trials × neurons
        X_pcx_test_session = X_pcx_test_session.T
        X_plcoa_test_session = X_plcoa_test_session.T

        # Match number of sampled trials
        N = min(
            X_pcx_test_session.shape[1],
            X_plcoa_test_session.shape[1],
            N
        )
    else:
        X_pcx_train = X_pcx
        X_plcoa_train = X_plcoa

    # Build training data (PCx)
    for sess in X_pcx_train:
        sess = sess.T  # trials × neurons
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:,idx])
        y.append(np.zeros(sess.shape[0]))

    # Build training data (plCoA)
    for sess in X_plcoa_train:
        sess = sess.T  # trials × neurons
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:,idx])
        y.append(np.ones(sess.shape[0]))

    # Build held-out test set
    if sess_index is not None:
        idx_pcx = rng.choice(X_pcx_test_session.shape[1], N, replace=False)
        idx_plcoa = rng.choice(X_plcoa_test_session.shape[1], N, replace=False)

        X_held = np.vstack([
            X_pcx_test_session[:,idx_pcx],
            X_plcoa_test_session[:,idx_plcoa]
        ])

        y_held = np.concatenate([
            np.zeros(X_pcx_test_session.shape[0]),
            np.ones(X_plcoa_test_session.shape[0])
        ])

        return np.vstack(X), np.concatenate(y), X_held, y_held

    return np.vstack(X), np.concatenate(y), None, None



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



def plot_mean_trial_correlation(sessions, title="Mean Trial–Trial Correlation"):
    corr_matrices = []
    
    for sess in sessions:
        sess = sess.T  # shape: (trials × neurons)
        corr = np.corrcoef(sess)
        corr_matrices.append(corr)
    
    # Pad with NaNs to match the largest trial count (if sessions differ)
    max_trials = max(c.shape[0] for c in corr_matrices)
    padded = []
    for c in corr_matrices:
        pad_size = max_trials - c.shape[0]
        padded_corr = np.full((max_trials, max_trials), np.nan)
        padded_corr[:c.shape[0], :c.shape[1]] = c
        padded.append(padded_corr)
    
    # Mean across sessions, ignoring NaNs
    mean_corr = np.nanmean(padded, axis=0)

    # Plot
    plt.imshow(mean_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title(title)
    plt.xlabel('Trial')
    plt.ylabel('Trial')
    plt.show()
    
def remove_highly_correlated_trials(sessions, top_k=10):
    """
    sessions: list of session matrices (neurons × trials)
    top_k: % of trials to remove (e.g. 10 = top 10% most correlated)
    Returns: list of trimmed session matrices (neurons × trials)
    """
    trimmed_sessions = []

    for sess in sessions:
        sess_T = sess.T  # trials × neurons
        n_trials = sess_T.shape[0]

        # how many trials to drop
        k = min(n_trials - 1, int(n_trials * 0.01 * top_k))

        corr = np.corrcoef(sess_T)
        np.fill_diagonal(corr, np.nan)
        mean_corr = np.nanmean(corr, axis=1)

        keep_idx = np.argsort(mean_corr)[:-k]
        trimmed = sess_T[keep_idx]

        trimmed_sessions.append(trimmed.T)  # back to neurons × trials

    return trimmed_sessions


# Remove highly correlated trials and check trial by trial correlation again
def filterAndPlotKpercent (X_pcx, X_plcoa, top_k=10,plot = True ):
    
    X_pcxFiltered = remove_highly_correlated_trials(X_pcx, top_k = top_k)
    X_plcoaFiltered = remove_highly_correlated_trials(X_plcoa, top_k=top_k)
    if plot:
        plot_mean_trial_correlation(X_pcxFiltered, title='PCx filtered trial by trial corr')
        plot_mean_trial_correlation(X_plcoaFiltered, title='plCoA filtered trial by trial corr')
    # Shuffle trial order
    pcxShuff = X_pcxFiltered.copy()
    plcoaShuff = X_plcoaFiltered.copy()
    for i in range(len(pcxShuff)):
        sess = pcxShuff[i].T  # shape: (trials, neurons)
        sess_shuffled = shuffle(sess, random_state=default_rng().integers(1e6))
        pcxShuff[i] = sess_shuffled.T  # back to (neurons, trials)

    for i in range(len(plcoaShuff)):
        sess = plcoaShuff[i].T
        sess_shuffled = shuffle(sess, random_state=default_rng().integers(1e6))
        plcoaShuff[i] = sess_shuffled.T
        
    if plot:
       
        plot_mean_trial_correlation(pcxShuff, title='PCx shuffled trial by trial corr')
        plot_mean_trial_correlation(plcoaShuff, title='plCoA shuffled trial by trial corr')
         
    return pcxShuff, plcoaShuff

# Check trial by trial correlation

X_pcx , X_plcoa = getRegionalData(data)
plot_mean_trial_correlation(X_pcx, title='PCx trial by trial corr')
plot_mean_trial_correlation(X_plcoa, title='plCoA trial by trial corr')
pcxShuff , plcoaShuff = filterAndPlotKpercent(X_pcx, X_plcoa)
print ("Just finished Checking trial by trail correlations")



"loop through the session, always leaving a differnt session for testing"

# 1 - Run build dataset builder for for each iteration with sess_index set to the loop index
# 2 - Run the classification function keeping the accuracies in a variable: "pseudoAcc" "trueAcc"
# 3 - Plot the accuracies as a distribution

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

print ("Just finished the comparison between pseudo population and held out sessions")

"Resample using variable N with all population"

# Optimising N for best performance
# Loop though building traing and testing to find the best N value : [5,10,15]

N_values = [5, 10, 15, 17, 20, 22, 25]
repeats_per_N = 200

accuracies_N_varies = []
num_trials_per_N = []

for N in N_values:
    N_accs = []
    trials = []
    
    for _ in range(repeats_per_N):
        out = build_dataset_with_holdout(X_pcx, X_plcoa, N, sess_index=None)
        if out is None:
            continue
        X, y, X_held, y_held = out
        
        trials.append(len(y))
        
        N_acc, _ = singleSVMClassification(X, y)
        N_accs.append(N_acc)
    
    # Store mean results across the 20 runs
    accuracies_N_varies.append(np.mean(N_accs))
    num_trials_per_N.append(np.mean(trials))


plt.figure(figsize=(6,4))
temp_accuraciesN_varied = np.array(accuracies_N_varies) * 100
plt.plot(N_values, temp_accuraciesN_varied, marker='*', linewidth=0, label='Pseudo-population')

plt.axhline(50, color='k', linestyle='--', linewidth=1, label='Chance')

plt.xlabel('Number of neurons (N)')
plt.ylabel('Classification accuracy %')
plt.xticks(N_values)
plt.title(f'Decoding accuracy vs subpopulation size {repeats_per_N} resamples')
plt.ylim(40, 100)
plt.legend()
plt.tight_layout()
plt.show()
print("Just finished the N sub popualation tuning curve")

"Using the optimal N to resample 100 times to get a full distribution with all data"
accuracies_dist = []
rng = np.random.default_rng()
for i in range(100):
    out  = build_dataset_with_holdout\
        (X_pcx,X_plcoa, 17 , sess_index = None,rng= rng)
    if out is None:
        continue
    X , y , X_held, y_held = out
    
    temp = singleSVMClassification(X, y)
    accuracies_dist.append(temp)


# Plotting

acc_full_resample = np.array([i[0] for i in accuracies_dist])
# histogram

plt.figure(figsize=(6,4))
plt.hist(acc_full_resample, bins=10, color='steelblue', edgecolor='black')
plt.axvline(0.5, color='k', linestyle='--')

plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Accuracy distribution across 100 resamples')
plt.tight_layout()
plt.show()
print ("Just finished the resampled classification distibution")

"Permutation test - building Null distribution"

if 'perm_means' not in locals():  # To not rerun permutations every debugging
    perm_means = []
    
    rng = np.random.default_rng(123)
    
    for p in range(50):
        perm_acc = []
        
        for i in range (50):
            out  = build_dataset_with_holdout\
                (X_pcx,X_plcoa, 17 , sess_index = None,rng= rng)
            if out is None:
                continue
            X , y , X_held, y_held = out
            
            y_perm = rng.permutation(y)
            
            P_acc = singleSVMClassification(X, y_perm)
            t = P_acc[0]
            
            perm_acc.append(t)
            
        perm_means.append(np.mean(perm_acc))
        perm_mean_acc = np.mean(perm_means)

print("Just finished building the null distribution with permutations")

"Building and training on Shuffled filterd data for classification"

filteredAccuracies_dist = []
rng = np.random.default_rng()
for i in range(100):
    out  = build_dataset_with_holdout\
        (pcxShuff,plcoaShuff, 17 , sess_index = None,rng= rng)
    if out is None:
        continue
    X , y , X_held, y_held = out
    
    temp = singleSVMClassification(X, y)
    filteredAccuracies_dist.append(temp)

# Plotting
acc_shuffled_dist = np.array([i[0] for i in filteredAccuracies_dist])
# histogram

plt.figure(figsize=(6,4))
plt.hist(acc_shuffled_dist, bins=10, color='steelblue', edgecolor='black')
plt.axvline(0.5, color='k', linestyle='--')

plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Accuracy distribution across shuffled filtered resamples')
plt.tight_layout()
plt.show()
print("Just finished the filtered shuffled data classification")

"Building top K% removed benchline"


K_values = [10,15,17,20,22,25,35,50,60,75,80,90]

accuracies_K_varies = []
num_trials_per_k = []
p_values_per_k = []

repeats_per_k = 750

for k in K_values:
    accs_k_curve = []
    trials = []
    pvals = []
    
    for _ in range(repeats_per_k):
        pcxShuff_bench, plcoaShuff_bench = filterAndPlotKpercent(X_pcx, X_plcoa, top_k=k, plot=False)
        out = build_dataset_with_holdout(pcxShuff_bench, plcoaShuff_bench, 17, sess_index=None)
        
        if out is None:
            continue
        
        X, y, X_held, y_held = out
        trials.append(len(y))
        
        acc_k_curve, _ = singleSVMClassification(X, y)
        accs_k_curve.append(acc_k_curve)
        
        # p-value from null distribution (perm_means must be defined earlier!)
        p = (1 + np.sum(perm_means >= acc_k_curve)) / (1 + len(perm_means))
        pvals.append(p)
    
    # Store mean results across the 20 runs
    accuracies_K_varies.append(np.mean(accs_k_curve))
    num_trials_per_k.append(np.mean(trials))
    p_values_per_k.append(np.mean(pvals))

    
    


# Plotting
kVaried_acc = np.array(accuracies_K_varies) * 100
num_trials = np.array(num_trials_per_k)
p_values = np.array(p_values_per_k)

plt.figure(figsize=(10, 5))
plt.plot(K_values, kVaried_acc, marker='o', linewidth=0, label='Pseudo-population')
plt.axhline(50, color='k', linestyle='--', linewidth=1, label='Chance')

# Annotate trial count below each point
for x, y_val, n in zip(K_values, kVaried_acc, num_trials):
    plt.text(x, y_val - 4, f'n={n}', ha='center', va='top', fontsize=6, color='gray')

# Annotate significance stars above points
for x, y_val, p in zip(K_values, kVaried_acc, p_values):
    if p < 0.05:
        plt.text(x, 90, '*', ha='center', va='bottom', fontsize=14, color='k',label = 'Significant')

plt.xlabel('Percentage of highest correlation removed')
plt.ylabel('Classification accuracy (%)')
plt.title(f'Decoding accuracy vs top correlation trial removal {repeats_per_k} resamples')
plt.xticks(K_values)
plt.ylim(40, 105)
plt.legend()
plt.tight_layout()
plt.show()
print ("Just finished the top K correlative trial removed tuning curve")

# Compute p-value

observed_mean = np.mean(acc_full_resample)

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


# Boxplot


# Convert to numpy arrays (important for plotting)
pseudoAcc = np.array(pseudoAcc)
perm_means = np.array(perm_means)
session_acc = np.array(trueAcc)
plt.figure(figsize=(6,5))

plt.boxplot(
    [session_acc, acc_full_resample, perm_means],
    tick_labels=[ 'Held-out session','Pseudo-population', 'Permutation'],
    patch_artist=True,
    boxprops=dict(facecolor='lightsteelblue'),
    medianprops=dict(color='black')
)

# Chance line
plt.axhline(0.5, color='k', linestyle='--', linewidth=1, label='Chance')

plt.ylabel('Classification accuracy')
plt.title('Decoding performance comparison')
plt.ylim(0.4, 1.0)
plt.legend()
plt.tight_layout()
plt.show()

print ("Just finished the accuracy comparison between the pseudo population, heldout sessions and pemutations")
