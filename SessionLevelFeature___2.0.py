#%% ============================================================
# 0) Imports + Config
# ==============================================================
import numpy as np
import h5py
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

from scipy.stats import skew

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity


# --- Paths (edit these) ---
PATH_MONO = "/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_HelperFunctions/region_sessions_Notnormalized.mat"
PATH_AA   = "/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_HelperFunctions/region_sessions_AA_Notnormalized.mat"
PATH_NAT  = "/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/FinalProject_HelperFunctions/region_sessions_Nat_Notnormalized.mat"

# --- Experiment knobs ---
SESSION_STRUCTURE = {"Mono": (15, 10), "Nat": (13, 10), "AA": (10, 10)}
N_SESS_PER_COND = 10
N_PERMUTATIONS = 550
EPS = 1e-8

# --- Debug / convenience knobs ---
SHOW_PLOTS = True      # set False to avoid plt.show blocking
PRINT_DEBUG = True
RANDOM_STATE = 42      # used in control split


#%% ============================================================
# 1) Loading helpers
# ==============================================================
def get_regional_sessions(mat):
    """Return lists of session matrices for PCx and plCoA from an HDF5 .mat file."""
    X_pcx, X_plcoa = [], []

    for ref in mat["x_pcx"]:
        X_pcx.append(np.array(mat[ref[0]]))

    for ref in mat["x_plcoa"]:
        X_plcoa.append(np.array(mat[ref[0]]))

    return X_pcx, X_plcoa


def load_all_sessions():
    """
    Loads Mono/Nat/AA datasets.

    Returns a dict:
      data["Mono"]["PCx"] -> list of session matrices
      data["Mono"]["plCoA"] -> list ...
      same for "Nat", "AA"
    """
    data = {"Mono": {}, "Nat": {}, "AA": {}}

    with h5py.File(PATH_MONO, "r") as f:
        pcx, pl = get_regional_sessions(f)
    data["Mono"]["PCx"] = pcx
    data["Mono"]["plCoA"] = pl

    with h5py.File(PATH_NAT, "r") as f:
        pcx, pl = get_regional_sessions(f)
    data["Nat"]["PCx"] = pcx
    data["Nat"]["plCoA"] = pl

    with h5py.File(PATH_AA, "r") as f:
        pcx, pl = get_regional_sessions(f)
    data["AA"]["PCx"] = pcx
    data["AA"]["plCoA"] = pl

    if PRINT_DEBUG:
        total = sum(len(data[c][r]) for c in data for r in data[c])
        print("Loaded total sessions:", total)
        for cond in ["Mono", "Nat", "AA"]:
            print(cond, "PCx:", len(data[cond]["PCx"]), "| plCoA:", len(data[cond]["plCoA"]))

    return data


#%% ============================================================
# 2) Normalization (per odor, per neuron)
# ==============================================================
def normalize_session_per_odor_per_neuron(session_data, structure, eps=EPS):
    """
    Per-odor, per-neuron z-score within a single session.
    Assumes trials are [odor1 reps | odor2 reps | ...].
    """
    n_odors, n_reps = structure
    n_neurons, n_trials = session_data.shape

    normalized = np.zeros_like(session_data, dtype=np.float64)

    for neuron_idx in range(n_neurons):
        neuron_trials = session_data[neuron_idx, :]
        for odor_idx in range(n_odors):
            start = odor_idx * n_reps
            end = start + n_reps

            if start >= n_trials:
                break
            end = min(end, n_trials)

            block = neuron_trials[start:end]
            mu = np.mean(block)
            sd = np.std(block)
            normalized[neuron_idx, start:end] = (block - mu) / (sd + eps)

    return normalized


#%% ============================================================
# 3) Feature functions
# ==============================================================
def mean_response(X): return float(np.mean(X))
def std_response(X):  return float(np.std(X))

def std_per_neuron(X):
    return float(np.mean(np.std(X, axis=1)))

def mean_peak_response(X):
    return float(np.mean(np.max(X, axis=1)))

def fraction_excited(X):
    m = np.mean(X, axis=1)
    return float(np.sum(m > 0) / m.size)

def fraction_suppressed(X):
    m = np.mean(X, axis=1)
    return float(np.sum(m < 0) / m.size)

def mean_trial_corr(X):
    if X.shape[1] < 2:
        return np.nan
    C = np.corrcoef(X.T)  # trials x trials
    mask = ~np.eye(C.shape[0], dtype=bool)
    return float(np.nanmean(C[mask]))

def trial_var_ratio(X):
    total = np.var(X)
    if total == 0:
        return 0.0
    per_neuron = np.var(X, axis=1)
    return float(np.mean(per_neuron) / total)

def neuronal_consistency(X):
    return float(np.mean(np.std(X, axis=1)))

def pairwise_neuron_corr_mean(X):
    if X.shape[0] < 2:
        return np.nan
    C = np.corrcoef(X)  # neurons x neurons
    mask = ~np.eye(C.shape[0], dtype=bool)
    return float(np.nanmean(C[mask]))

def pc1_explained_var(X):
    if min(X.shape) < 2:
        return 0.0
    pca = PCA(n_components=1)
    pca.fit(X)
    return float(pca.explained_variance_ratio_[0])

def dimensionality_ratio(X, threshold=0.9):
    n_components = min(X.shape)
    if n_components < 1:
        return 0.0
    pca = PCA(n_components=n_components)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cum, threshold) + 1)
    return float(n_pc / X.shape[0])

def participation_ratio(X):
    if X.shape[0] < 2:
        return 0.0
    cov = np.cov(X)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-10]
    if eig.size == 0:
        return 0.0
    return float((np.sum(eig) ** 2) / np.sum(eig ** 2))

def extract_features(X):
    """One place to edit your feature vector."""
    return [
        mean_response(X),
        std_response(X),
        std_per_neuron(X),
        mean_peak_response(X),
        fraction_excited(X),
        fraction_suppressed(X),
        mean_trial_corr(X),
        trial_var_ratio(X),
        neuronal_consistency(X),
        pairwise_neuron_corr_mean(X),
        pc1_explained_var(X),
        dimensionality_ratio(X),
        participation_ratio(X),
    ]


#%% ============================================================
# 4) Build REAL dataset (PCx vs plCoA)
# ==============================================================
def build_real_dataset(data, n_per_cond=N_SESS_PER_COND):
    """
    Builds X_all, y_all from first n_per_cond sessions in each condition for each region.
    Returns:
      X_all: (n_sessions x n_features)
      y_all: list of labels ("PCx" or "plCoA")
      meta:  list of tuples (region, condition, n_neurons)
    """
    X_all, y_all, meta = [], [], []

    for region in ["plCoA", "PCx"]:
        for cond in ["Mono", "Nat", "AA"]:
            sess_list = data[cond][region][:n_per_cond]
            for sess in sess_list:
                Xn = normalize_session_per_odor_per_neuron(sess, SESSION_STRUCTURE[cond])
                X_all.append(extract_features(Xn))
                y_all.append(region)
                meta.append((region, cond, sess.shape[0]))

    X_all = np.array(X_all, dtype=float)

    if PRINT_DEBUG:
        print("Real dataset:", X_all.shape[0], "sessions x", X_all.shape[1], "features")
        print("Label counts:", Counter(y_all))
        print("Neuron totals:",
              {r: sum(n for (rr, _, n) in meta if rr == r) for r in ["PCx", "plCoA"]})

    return X_all, y_all, meta


#%% ============================================================
# 5) Control dataset (within-region fake split)
# ==============================================================
def split_sessions_by_neuron_count(sessions, seed=RANDOM_STATE):
    """
    Shuffle sessions then split into two groups so total neuron counts are ~balanced.
    Returns group0, group1 (lists of tuples: (cond, session_matrix))
    """
    sessions = shuffle(sessions, random_state=seed)

    neuron_counts = [sess.shape[0] for (_, sess) in sessions]
    target = np.floor(np.sum(neuron_counts) / 2)

    cum = 0
    cutoff = 0
    for i, n in enumerate(neuron_counts):
        cum += n
        if cum >= target:
            cutoff = i + 1
            break

    return sessions[:cutoff], sessions[cutoff:]


def build_control_dataset(data, region="PCx", seed=RANDOM_STATE):
    """
    Within one region: assign fake labels 0/1 to two halves.
    Returns:
      X_ctrl, y_ctrl, debug_info
    """
    # pool all conditions for that region
    pooled = []
    for cond in ["Mono", "Nat", "AA"]:
        for sess in data[cond][region]:
            pooled.append((cond, sess))

    g0, g1 = split_sessions_by_neuron_count(pooled, seed=seed)

    def to_features(group, fake_label):
        X, y, meta = [], [], []
        for cond, sess in group:
            Xn = normalize_session_per_odor_per_neuron(sess, SESSION_STRUCTURE[cond])
            X.append(extract_features(Xn))
            y.append(fake_label)
            meta.append((cond, sess.shape[0]))
        return X, y, meta

    X0, y0, meta0 = to_features(g0, "0")
    X1, y1, meta1 = to_features(g1, "1")

    X_ctrl = np.array(X0 + X1, dtype=float)
    y_ctrl = y0 + y1

    debug_info = {
        "group0_conditions": Counter([c for (c, _) in meta0]),
        "group1_conditions": Counter([c for (c, _) in meta1]),
        "group0_neurons": sum(n for (_, n) in meta0),
        "group1_neurons": sum(n for (_, n) in meta1),
        "n_sessions0": len(meta0),
        "n_sessions1": len(meta1),
    }

    if PRINT_DEBUG:
        print(f"\nCONTROL DEBUG ({region}, seed={seed})")
        print(" group0:", debug_info["n_sessions0"], "sessions | neurons:", debug_info["group0_neurons"],
              "| conds:", debug_info["group0_conditions"])
        print(" group1:", debug_info["n_sessions1"], "sessions | neurons:", debug_info["group1_neurons"],
              "| conds:", debug_info["group1_conditions"])

    return X_ctrl, y_ctrl, debug_info


#%% ============================================================
# 6) LOOCV + permutation
# ==============================================================
def loocv_accuracy(X, y, clf=None):
    if clf is None:
        clf = SVC(kernel="linear", C=1)

    loo = LeaveOneOut()
    y = np.array(y)

    preds = []
    truths = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]

        preds.append(pred)
        truths.append(y_test[0])

    acc = accuracy_score(truths, preds)
    return preds, truths, float(acc)


def permutation_test(X, y, n_perm=N_PERMUTATIONS):
    real_preds, real_truths, real_acc = loocv_accuracy(X, y)
    null = []
    for _ in range(n_perm):
        y_shuf = shuffle(y, random_state=None)
        _, _, acc = loocv_accuracy(X, y_shuf)
        null.append(acc)

    p = (1 + sum(a >= real_acc for a in null)) / (1 + len(null))
    return real_acc, null, p, real_preds, real_truths


#%% ============================================================
# 7) Plot helpers (safe)
# ==============================================================
def plot_null(null_accs, real_acc, title="Permutation test"):
    plt.figure(figsize=(6,4), dpi=200)
    plt.hist(null_accs, bins=20, alpha=0.7)
    plt.axvline(real_acc, linestyle="--")
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.xlim(0, 1)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


#%% ============================================================
# 8) RUN PIPELINE (run sections manually)
# ==============================================================
# 8.1 Load data
data = load_all_sessions()

# 8.2 Build real dataset + run LOOCV
X_all, y_all, meta = build_real_dataset(data, n_per_cond=N_SESS_PER_COND)
preds, truths, acc = loocv_accuracy(X_all, y_all)
print("\nREAL LOOCV acc:", acc)
print("REAL confusion:\n", confusion_matrix(truths, preds, labels=["PCx", "plCoA"]))

# 8.3 Controls (run one seed)
X_ctrl_pcx, y_ctrl_pcx, dbg_pcx = build_control_dataset(data, region="PCx", seed=RANDOM_STATE)
preds_c, truths_c, acc_c = loocv_accuracy(X_ctrl_pcx, y_ctrl_pcx)
print("\nCONTROL PCx acc:", acc_c)
print("CONTROL PCx confusion:\n", confusion_matrix(truths_c, preds_c, labels=["0", "1"]))



# 8.4 Control distribution (many seeds) — this is the diagnostic you wanted
accs_pcx = []
for seed in range(300):
    Xc, yc, _ = build_control_dataset(data, region="PCx", seed=seed)
    _, _, a = loocv_accuracy(Xc, yc)
    accs_pcx.append(a)
accs_pcx = np.array(accs_pcx)
print("\nPCx control distribution: mean =", accs_pcx.mean(), "std =", accs_pcx.std())
if SHOW_PLOTS:
    plt.figure(figsize=(6,4), dpi=200)
    plt.hist(accs_pcx, bins=20, alpha=0.7)
    plt.axvline(0.5, linestyle="--")
    plt.title("PCx control accuracy across seeds")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()

# 8.3.1
X_ctrl_plcoa, y_ctrl_plcoa, dbg_plcoa = build_control_dataset(data, region="plCoA", seed=RANDOM_STATE)
preds_c, truths_c, acc_c = loocv_accuracy(X_ctrl_plcoa, y_ctrl_plcoa)
print("\nCONTROL plCoA acc:", acc_c)
print("CONTROL plCoA confusion:\n", confusion_matrix(truths_c, preds_c, labels=["0", "1"]))

# 8.4.1 Control distribution (many seeds) — this is the diagnostic you wanted
accs_plcoa = []
for seed in range(300):
    Xc, yc, _ = build_control_dataset(data, region="plCoA", seed=seed)
    _, _, a = loocv_accuracy(Xc, yc)
    accs_plcoa.append(a)
accs_plcoa = np.array(accs_plcoa)
print("\nplCoA control distribution: mean =", accs_plcoa.mean(), "std =", accs_plcoa.std())
if SHOW_PLOTS:
    plt.figure(figsize=(6,4), dpi=200)
    plt.hist(accs_plcoa, bins=20, alpha=0.7)
    plt.axvline(0.5, linestyle="--")
    plt.title("plCoA control accuracy across seeds")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()

# 8.5 Permutation test on real labels
real_acc, null_accs, p, real_preds, real_truths = permutation_test(X_all, y_all, n_perm=N_PERMUTATIONS)
print("\nPermutation p =", p, "| real acc =", real_acc)
plot_null(null_accs, real_acc, title=f"Real acc={real_acc:.3f}, p={p:.4f}")
