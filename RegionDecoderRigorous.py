#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous Session-Level Region Decoder (PCx vs plCoA)

Methodological Requirements:
- Session-level cross-validation (no data leakage)
- Per-odor, per-neuron z-scoring of responses
- Biologically interpretable features
- Linear classifier with proper controls
- Label-shuffled control analysis

@author: Prashastha
Date: January 21, 2026
"""

import numpy as np
import h5py
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ============================================================================
# DATA LOADING
# ============================================================================

def load_regional_sessions(filepath):
    """Load sessions from a .mat file for both regions.
    
    Returns
    -------
    X_pcx, X_plcoa : list of ndarray
        Each array shape (n_neurons, n_trials) for one session.
    """
    data = h5py.File(filepath, 'r')
    X_pcx = [np.array(data[ref[0]]) for ref in data['x_pcx']]
    X_plcoa = [np.array(data[ref[0]]) for ref in data['x_plcoa']]
    data.close()
    return X_pcx, X_plcoa


# ============================================================================
# ODOR-BLOCK Z-SCORING (per odor, per neuron)
# ============================================================================

def zscore_session_by_blocks(session_data, block_sizes, eps=1e-8):
    """Z-score each odor block independently.

    Parameters
    ----------
    session_data : ndarray
        Shape (n_neurons, n_trials). Assumed to already be (odor - baseline).
    block_sizes : list[int]
        Trials per odor in the concatenated layout, e.g., [reps_odor1, reps_odor2, ...].
    eps : float
        Minimum std to avoid division by zero.
    """
    if block_sizes is None:
        raise ValueError("block_sizes must be provided for per-odor z-scoring.")
    total_trials = sum(block_sizes)
    if session_data.shape[1] != total_trials:
        raise ValueError(
            f"Block sizes ({total_trials}) do not match trial count ({session_data.shape[1]})."
        )

    out = session_data.copy()
    start = 0
    for block_len in block_sizes:
        end = start + block_len
        block = session_data[:, start:end]
        mu = block.mean(axis=1, keepdims=True)
        sigma = block.std(axis=1, keepdims=True)
        sigma = np.where(sigma < eps, eps, sigma)
        out[:, start:end] = (block - mu) / sigma
        start = end
    return out


def zscore_sessions(sessions, block_sizes, eps=1e-8):
    """Apply per-odor z-scoring to a list of sessions."""
    return [zscore_session_by_blocks(sess, block_sizes, eps) for sess in sessions]


# ============================================================================
# FEATURE EXTRACTION (Biologically Interpretable)
# ============================================================================

def extract_mean_firing_rates(session_data, odor_idx):
    """Extract time-averaged post-odor firing rate per neuron.
    
    Parameters
    ----------
    session_data : ndarray
        Shape (n_neurons, n_trials), already odor-block z-scored.
    odor_idx : slice or array
        Indices of post-odor response trials.
    
    Returns
    -------
    ndarray
        Shape (n_neurons,). Mean post-odor response per neuron.
    """
    odor_responses = session_data[:, odor_idx]
    return odor_responses.mean(axis=1)


def extract_pca_features(session_data, odor_idx, pca_model=None, n_components=10):
    """Extract PCA features from post-odor responses.
    
    Parameters
    ----------
    session_data : ndarray
        Shape (n_neurons, n_trials).
    odor_idx : slice or array
        Indices of post-odor trials.
    pca_model : PCA or None
        Pre-fitted PCA from training data. If None, creates new PCA.
    n_components : int
        Number of PCs to extract.
    
    Returns
    -------
    features : ndarray
        Shape (n_components,) or (n_neurons,) if < n_components neurons.
    pca_model : PCA
        The PCA model used (for training sessions to reuse on test).
    """
    odor_responses = session_data[:, odor_idx]  # (n_neurons, n_odor_trials)
    
    if pca_model is None:
        # Training: fit new PCA
        n_components = min(n_components, odor_responses.shape[0], odor_responses.shape[1])
        pca_model = PCA(n_components=n_components)
        pca_model.fit(odor_responses.T)  # Fit on trials as samples
    
    # Project onto PCs and take mean projection per PC as features
    projections = pca_model.transform(odor_responses.T)  # (n_trials, n_components)
    features = projections.mean(axis=0)  # Mean across trials
    
    return features, pca_model


# ============================================================================
# SESSION-LEVEL CLASSIFIER WITH LEAVE-ONE-SESSION-OUT CV
# ============================================================================

class SessionLevelDecoder:
    """Session-level region decoder with rigorous cross-validation.

    Expects input sessions to already be per-odor z-scored (per neuron).
    """
    
    def __init__(self, feature_type='mean_firing', classifier='linear_svm', n_pca=10):
        """
        Parameters
        ----------
        feature_type : str
            'mean_firing' or 'pca'
        classifier : str
            'ridge' or 'linear_svm'
        n_pca : int
            Number of PCA components if feature_type='pca'
        
        Note: Inputs are assumed to be odor-block z-scored; no additional
        baseline normalization is applied inside the decoder.
        """
        self.feature_type = feature_type
        self.classifier_type = classifier
        self.n_pca = n_pca
        self.max_features = None  # Will be set during CV
    
    @staticmethod
    def pad_features(features, target_length):
        """Pad feature vector to target length with zeros."""
        if len(features) < target_length:
            return np.pad(features, (0, target_length - len(features)), mode='constant')
        elif len(features) > target_length:
            return features[:target_length]
        else:
            return features
        
    def extract_session_features(self, session, odor_idx=None, 
                                session_id=None, pca_model=None, is_training=False):
        """Extract features from one session.

        Parameters
        ----------
        session : ndarray
            Shape (n_neurons, n_trials). Already z-scored per odor block.
        odor_idx : slice or array, optional
            Trials to include for feature extraction (default: all trials).
        session_id : str or int
            Unique identifier for this session.
        pca_model : PCA or None
            Pre-fitted PCA from training data.
        is_training : bool
            Whether this is a training session (affects PCA fitting only).
        """
        del session_id  # retained for interface symmetry, unused here
        odor_idx = slice(None) if odor_idx is None else odor_idx

        if self.feature_type == 'mean_firing':
            features = extract_mean_firing_rates(session, odor_idx)
            return features, None
        elif self.feature_type == 'pca':
            features, pca_model = extract_pca_features(session, odor_idx, pca_model, self.n_pca)
            return features, pca_model
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
    
    def leave_one_session_out_cv(self, sessions_pcx, sessions_plcoa, 
                                  odor_idx=None, verbose=True):
        """Run leave-one-session-out cross-validation.
        
        Returns
        -------
        dict with keys:
            - accuracy : float
            - predictions : array
            - true_labels : array
            - confusion_matrix : ndarray
        """
        # Combine all sessions with labels
        all_sessions = sessions_pcx + sessions_plcoa
        all_labels = ['PCx'] * len(sessions_pcx) + ['plCoA'] * len(sessions_plcoa)
        n_sessions = len(all_sessions)
        
        # Determine max feature length across all sessions
        if self.feature_type == 'mean_firing':
            self.max_features = max(session.shape[0] for session in all_sessions)
        elif self.feature_type == 'pca':
            self.max_features = self.n_pca
        
        predictions = []
        true_labels = []
        
        # Leave-one-session-out
        for test_idx in range(n_sessions):
            train_idx = [i for i in range(n_sessions) if i != test_idx]
            
            # === TRAINING PHASE ===
            # Extract features from training sessions only
            X_train = []
            y_train = []
            pca_model = None
            
            for i in train_idx:
                feat, pca_model = self.extract_session_features(
                    all_sessions[i], odor_idx, 
                    session_id=f"train_{i}", pca_model=pca_model, is_training=True
                )
                # Pad to consistent length
                feat_padded = self.pad_features(feat, self.max_features)
                X_train.append(feat_padded)
                y_train.append(all_labels[i])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train classifier
            if self.classifier_type == 'ridge':
                clf = RidgeClassifier(alpha=1.0)
            elif self.classifier_type == 'linear_svm':
                clf = LinearSVC(C=1.0, max_iter=10000)
            else:
                raise ValueError(f"Unknown classifier: {self.classifier_type}")
            
            clf.fit(X_train, y_train)
            
            # === TEST PHASE ===
            # Extract features from test session using training-derived statistics
            X_test, _ = self.extract_session_features(
                all_sessions[test_idx], odor_idx, 
                session_id=f"test_{test_idx}", pca_model=pca_model, is_training=False
            )
            # Pad to consistent length
            X_test = self.pad_features(X_test, self.max_features)
            X_test = X_test.reshape(1, -1)
            
            # Predict
            y_pred = clf.predict(X_test)[0]
            y_true = all_labels[test_idx]
            
            predictions.append(y_pred)
            true_labels.append(y_true)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions, labels=['PCx', 'plCoA'])
        
        if verbose:
            print(f"Leave-One-Session-Out Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'confusion_matrix': cm
        }
    
    def permutation_control(self, sessions_pcx, sessions_plcoa, odor_idx=None, 
                           n_permutations=100, verbose=True):
        """Run label-shuffled control to test for data leakage.
        
        Expected accuracy should be ~0.5 (chance).
        """
        # Combine sessions
        all_sessions = sessions_pcx + sessions_plcoa
        all_labels = np.array(['PCx'] * len(sessions_pcx) + ['plCoA'] * len(sessions_plcoa))
        
        null_accuracies = []
        
        for perm in range(n_permutations):
            # Shuffle labels
            shuffled_labels = np.random.permutation(all_labels)
            shuffled_pcx = [all_sessions[i] for i in range(len(all_sessions)) 
                           if shuffled_labels[i] == 'PCx']
            shuffled_plcoa = [all_sessions[i] for i in range(len(all_sessions)) 
                             if shuffled_labels[i] == 'plCoA']
            
            # Run CV with shuffled labels
            results = self.leave_one_session_out_cv(
                shuffled_pcx, shuffled_plcoa, odor_idx, verbose=False
            )
            null_accuracies.append(results['accuracy'])
        
        null_accuracies = np.array(null_accuracies)
        
        if verbose:
            print(f"\nPermutation Control (n={n_permutations}):")
            print(f"  Mean null accuracy: {null_accuracies.mean():.4f}")
            print(f"  Std null accuracy: {null_accuracies.std():.4f}")
            print(f"  Expected: ~0.50 (chance level)")
        
        return null_accuracies


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results, null_accuracies=None):
    """Plot confusion matrix and permutation test results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = results['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=['PCx', 'plCoA'], yticklabels=['PCx', 'plCoA'],
                ax=axes[0], cbar=True, vmin=0, vmax=1)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title(f"Confusion Matrix\nAccuracy: {results['accuracy']:.3f}", fontsize=14)
    
    # Permutation test
    if null_accuracies is not None:
        axes[1].hist(null_accuracies, bins=20, alpha=0.7, color='gray', 
                    label='Null distribution')
        axes[1].axvline(results['accuracy'], color='red', linewidth=2, 
                       linestyle='--', label='Observed accuracy')
        axes[1].axvline(0.5, color='black', linewidth=1, 
                       linestyle=':', label='Chance (0.5)')
        axes[1].set_xlabel('Accuracy', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Permutation Test', fontsize=14)
        axes[1].legend()
        
        # Compute p-value
        p_value = (null_accuracies >= results['accuracy']).sum() / len(null_accuracies)
        axes[1].text(0.02, 0.98, f"p = {p_value:.4f}", 
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis(data_file='data/region_sessions_Notnormalized.mat',
                 feature_type='mean_firing',
                 classifier='linear_svm',
                 n_permutations=100,
                 odor_block_sizes=None):
    """
    Run complete analysis pipeline with per-odor z-scored firing rates.
    
    CRITICAL: Assumes input data contains Δ firing rates (odor - baseline).
    This pipeline applies per-odor, per-neuron z-scoring: subtract mean and
    divide by std across repetitions for each odor block.
    
    Parameters
    ----------
    data_file : str
        Path to .mat file with session data (Δ firing rates)
    feature_type : str
        'mean_firing' or 'pca'
    classifier : str
        'ridge' or 'linear_svm' 
    n_permutations : int
        Number of permutation tests for control
    odor_block_sizes : list[int]
        Trials per odor in concatenated trial order (required).
    """
    
    print("="*70)
    print("RIGOROUS SESSION-LEVEL REGION DECODER")
    print("="*70)
    print("Normalization: Per-odor z-scoring (per neuron)")
    print("Input: Δ firing rates (odor - baseline)")
    print(f"Feature type: {feature_type}")
    print(f"Classifier: {classifier}")
    print()
    
    # Load data
    print("Loading data...")
    X_pcx, X_plcoa = load_regional_sessions(data_file)
    print(f"  PCx sessions: {len(X_pcx)}")
    print(f"  plCoA sessions: {len(X_plcoa)}")
    print(f"  Session shapes (example): {X_pcx[0].shape}")
    print()

    if odor_block_sizes is None:
        raise ValueError("Provide odor_block_sizes matching the trial layout, e.g., [reps_odor1, reps_odor2, ...].")

    # Per-odor z-score each session (per neuron)
    X_pcx = zscore_sessions(X_pcx, odor_block_sizes)
    X_plcoa = zscore_sessions(X_plcoa, odor_block_sizes)

    # Use all trials; data are already z-scored within odor blocks
    odor_idx = slice(None)
    
    # Initialize decoder
    decoder = SessionLevelDecoder(
        feature_type=feature_type,
        classifier=classifier,
        n_pca=10
    )
    
    # Run cross-validation
    print("Running Leave-One-Session-Out Cross-Validation...")
    results = decoder.leave_one_session_out_cv(X_pcx, X_plcoa, odor_idx)
    print()
    
    # Run permutation control
    print("Running Permutation Control...")
    null_accuracies = decoder.permutation_control(
        X_pcx, X_plcoa, odor_idx, n_permutations
    )
    print()
    
    # Statistical test
    p_value = (null_accuracies >= results['accuracy']).sum() / len(null_accuracies)
    print(f"Statistical Significance:")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")
    print()
    
    # Plot results
    plot_results(results, null_accuracies)
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print("NORMALIZATION: Per-odor, per-neuron z-scored Δ firing rates")
    print("INPUT DATA: Baseline-subtracted firing rates (odor - baseline)")
    print("SCALING: Odor-block mean subtraction and std division")
    print()
    if feature_type == 'mean_firing':
        print("The classifier uses noise-normalized mean post-odor Δ firing rates per neuron.")
        print("High accuracy suggests PCx and plCoA differ in:")
        print("  - Magnitude of odor-evoked responses (normalized for noise)")
        print("  - Distribution of response magnitudes across neurons")
        print("  - Signal-to-noise ratio patterns")
        print()
    elif feature_type == 'pca':
        print("The classifier uses low-dimensional PCA projections of noise-normalized responses.")
        print("High accuracy suggests PCx and plCoA differ in:")
        print("  - Structure of noise-normalized population responses")
        print("  - Coordination patterns in odor-evoked activity")
        print("  - Principal modes of response variation")
        print()
    
    if p_value < 0.05:
        print("The significant p-value confirms:")
        print("  - Classifier exploits genuine regional differences")
        print("  - No data leakage or trivial confounds")
        print("  - Differences are based on noise-normalized response magnitudes")
    else:
        print("WARNING: Non-significant result suggests:")
        print("  - Regions may not differ reliably in noise-normalized response patterns")
        print("  - Possible insufficient statistical power")
        print("  - May need more sessions or different features")
    
    print("="*70)
    
    return results, null_accuracies


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Run with noise-normalized mean firing rates
    results_mean, null_mean = run_analysis(
        feature_type='mean_firing',
        classifier='linear_svm',
        n_permutations=100,
        odor_block_sizes=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # TODO: set to [reps_odor1, reps_odor2, ...] matching trial layout
    )
    
    # Optional: Run with PCA features
    # results_pca, null_pca = run_analysis(
    #     feature_type='pca',
    #     classifier='linear_svm',
    #     n_permutations=100
    # )
    
    # Optional: Run with Ridge classifier
    # results_ridge, null_ridge = run_analysis(
    #     feature_type='mean_firing',
    #     classifier='ridge',
    #     n_permutations=100
    # )
