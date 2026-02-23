# Neural Activity Classification: PCx vs plCoA

This project implements machine learning models to classify neural activity data from two brain regions: **Piriform Cortex (PCx)** and **Posterolateral Cortical Amygdaloid Area (plCoA)** using data from Iurilli & Datta 2017.

## Project Overview

The goal is to build classifiers that can distinguish between neural activity patterns from these two olfactory brain regions using:
- **Traditional ML**: Support Vector Machines (SVM)
- **Deep Learning**: Graph Neural Networks (GNN) that model neurons as nodes and their correlations as edges

## Project Structure

```
IurilliAndDatta2017/
├── data/                           # Neural activity datasets (.mat files)
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── intermediate_report.ipynb   # Session-level analysis
│   ├── initial_report.ipynb        # Data exploration
│   └── data_preprocessor.ipynb     # Data preprocessing pipeline
├── SessionLevelFeaturesNonNormalizedGNN.py  # Main GNN implementation
├── SessionLevelFeaturesNonNormalized.py     # SVM baseline
├── SessionLevelFeaturesNonNormalizedFast.py # Fast SVM version
├── RegionDecoderRigorous.py        # Rigorous cross-validation decoder
├── functions.py                    # Utility functions
├── dataset.py                      # Data loading utilities
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd IurilliAndDatta2017

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Structure

The project uses three experimental conditions with different numbers of odors:

| Condition | Odors | Trials per Odor | Total Trials per Session |
|-----------|-------|-----------------|--------------------------|
| **Mono**  | 15    | 10              | 150                      |
| **Nat**   | 13    | 10              | 130                      |
| **AA**    | 10    | 10              | 100                      |

Each session contains:
- **Neurons**: Variable number per session
- **Trials**: Neural firing rates per odor presentation
- **Regions**: PCx or plCoA labels

### 3. Run the Models

#### Graph Neural Network (Recommended)

```bash
python SessionLevelFeaturesNonNormalizedGNN.py
```

This will:
1. Load data from all three conditions (Mono, Nat, AA)
2. Create graphs where neurons are nodes connected by correlation-based edges
3. Train a GNN with proper train/test split
4. Display training progress, confusion matrices, and final accuracy

#### Support Vector Machine Baseline

```bash
# Standard SVM with RBF kernel
python SessionLevelFeaturesNonNormalized.py

# Fast SVM with linear kernel  
python SessionLevelFeaturesNonNormalizedFast.py
```

#### Interactive Analysis

```bash
jupyter notebook
# Navigate to notebooks/ folder and open desired notebook
```

## Model Details

### Graph Neural Network Architecture

- **Nodes**: Individual neurons with 5 statistical features (mean, std, skewness, max, min firing rates)
- **Edges**: Based on correlation between neuron firing patterns (threshold: 0.3)
- **Layers**: 
  - 3 Graph Convolutional layers with residual connections
  - Graph Attention layer (4 heads)
  - Global mean pooling for session-level representation
  - Classification head (2 classes: PCx vs plCoA)

### Data Preprocessing

1. **Per-odor, per-neuron z-scoring**: Each neuron's response to each odor is z-scored independently
2. **Session-level features**: Statistical measures computed across all trials per neuron
3. **Graph construction**: Correlation matrices converted to adjacency matrices

## Expected Results

- **GNN Performance**: ~79% accuracy with proper generalization
- **SVM Baseline**: ~70% accuracy on linear Kernel (RBF showed no significant improvement)
- **Cross-validation**: Leave-one-session-out validation ensures robust evaluation

## Understanding the Output

### Training Progress
```
Epoch 1/1000, Loss: 0.6916, Training Accuracy: 54.55%
Epoch 21/1000, Loss: 0.6853, Training Accuracy: 54.55%
...
```

### Final Results
```
Held-out Test Accuracy: 78.57%
```

Plus confusion matrices and training curves showing model performance.

## Key Features

- **Multi-condition Training**: Uses all experimental conditions for better generalization
- **Proper Data Splits**: Clean train/test separation to prevent overfitting
- **Graph-based Modeling**: Captures neural connectivity patterns
- **Statistical Robustness**: Includes permutation tests and cross-validation
- **Visualization**: Training curves, confusion matrices, and performance metrics

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- `torch` + `torch-geometric` (GNN implementation)
- `scikit-learn` (SVM and metrics)
- `numpy`, `pandas`, `matplotlib` (data handling and visualization)
- `h5py` (loading .mat files)

## Contributing

1. Experiment with different GNN architectures in `NeuralGNN` class
2. Try different correlation thresholds for graph construction
3. Add new feature extraction methods in `create_graph_from_session()`
4. Implement additional baseline models

## Citation

Based on data from:
> Iurilli, G., & Datta, S. R. (2017). Population coding in an innately relevant olfactory area. *Neuron*, 93(5), 1180-1197.

## Troubleshooting

**OutOfMemoryError**: Reduce batch size in `train_gnn_model()` function
**Import Errors**: Ensure all dependencies installed via `pip install -r requirements.txt`
**CUDA Issues**: Model automatically detects and uses CPU if CUDA unavailable