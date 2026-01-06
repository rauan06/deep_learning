# Deep Learning Homework 1 - README

## Overview
This repository contains the complete solution for Deep Learning Homework 1, covering:
- **Problem 1**: Implementing and Training MLPs (8 points)
- **Problem 2**: Optimization and Training Dynamics (9 points)
- **Problem 3**: Convolutional Neural Networks (8 points)

**Total Points**: 25

---

## 📋 Requirements

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

### Installation
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU**: Recommended (CUDA-enabled GPU) but not required
- **RAM**: Minimum 8GB
- **Storage**: ~2GB for datasets (MNIST + CIFAR-10)

---

## 🚀 Quick Start

### Running the Complete Solution
```bash
python dl_hw1_solution.py
```

This will:
1. Download MNIST and CIFAR-10 datasets automatically
2. Train all models across all problems
3. Generate 9 visualization plots
4. Print results and analysis to console

**Expected Runtime**:
- CPU only: ~45-60 minutes
- GPU (CUDA): ~15-20 minutes

---

## 📁 Project Structure

```
.
├── dl_hw1_solution.py              # Main solution script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/                           # Auto-created dataset directory
│   ├── MNIST/
│   └── CIFAR-10/
└── outputs/                        # Generated plots (auto-created)
    ├── problem_1b_activation_comparison.png
    ├── problem_1c_xor_boundary.png
    ├── problem_2a_optimizer_comparison.png
    ├── problem_2b_gradient_analysis.png
    ├── problem_2c_regularization.png
    ├── problem_3a_cnn_training.png
    ├── problem_3a_confusion_matrix.png
    ├── problem_3c_filters.png
    └── problem_3c_activations.png
```

---

## 📊 Problems & Solutions

### Problem 1: Implementing and Training MLPs (8 points)

#### Part A: MLP Implementation (3 points)
- **Architecture**: 784 → 128 → 64 → 10
- **Dataset**: MNIST
- **Features**: Flexible MLP class supporting multiple architectures and activations
- **Output**: Training/test accuracy printed to console

#### Part B: Activation Function Comparison (3 points)
- **Activations Tested**: ReLU, Sigmoid, Tanh
- **Output**: `problem_1b_activation_comparison.png`
- **Analysis**: ReLU typically performs best due to reduced vanishing gradient issues

#### Part C: Solving XOR (2 points)
- **Architecture**: 2 → 4 → 1
- **Output**: 
  - XOR accuracy (should be ~100%)
  - Decision boundary visualization: `problem_1c_xor_boundary.png`

---

### Problem 2: Optimization and Training Dynamics (9 points)

#### Part A: Optimizer Comparison (4 points)
- **Optimizers**: SGD, SGD+Momentum, RMSprop, Adam
- **Epochs**: 20
- **Output**: `problem_2a_optimizer_comparison.png`
- **Metrics**: Training loss, test accuracy, training time

**Key Findings**:
- Adam typically converges fastest
- SGD with momentum shows steady improvement
- RMSprop balances speed and stability

#### Part B: Gradient Analysis (3 points)
- **Networks**: Deep MLP (6 layers) with Sigmoid vs ReLU
- **Output**: `problem_2b_gradient_analysis.png`
- **Visualization**: Log-scale gradient magnitudes across layers

**Key Findings**:
- Sigmoid suffers from vanishing gradients in deeper layers
- ReLU maintains gradient flow better

#### Part C: Regularization Effects (2 points)
- **Methods**: No regularization, L2 (weight_decay=0.001), Dropout (p=0.5)
- **Output**: `problem_2c_regularization.png`
- **Comparison**: Train vs test accuracy for each method

**Key Findings**:
- Dropout typically prevents overfitting most effectively
- L2 regularization provides subtle improvement
- Gap between train/test accuracy indicates overfitting

---

### Problem 3: Convolutional Neural Networks (8 points)

#### Part A: Basic CNN Implementation (4 points)
- **Architecture**:
  - Conv1: 32 filters, 3×3, ReLU, same padding → MaxPool 2×2
  - Conv2: 64 filters, 3×3, ReLU, same padding → MaxPool 2×2
  - Flatten → FC: 128 → Output: 10
- **Dataset**: CIFAR-10
- **Epochs**: 20
- **Outputs**:
  - Training/validation curves: `problem_3a_cnn_training.png`
  - Confusion matrix: `problem_3a_confusion_matrix.png`

**Expected Performance**: ~70-75% test accuracy

#### Part B: Architecture Experimentation (3 points)
**Status**: Not implemented in base script

To add custom architectures, modify the `BasicCNN` class and compare:
- Number of convolutional layers
- Filter counts
- Kernel sizes
- Pooling strategies (max vs average)

#### Part C: Visualizing Learned Features (1 point)
- **Outputs**:
  - First layer filters (8 filters): `problem_3c_filters.png`
  - Activation maps (3 samples): `problem_3c_activations.png`

**Observations**:
- Early filters detect edges, colors, and simple patterns
- Deeper layers learn more complex features

---

## 🎯 Results Summary

### Expected Accuracies

| Problem | Model | Expected Accuracy |
|---------|-------|-------------------|
| 1A | MLP (ReLU) | ~97-98% |
| 1B | ReLU | ~97-98% |
| 1B | Sigmoid | ~95-97% |
| 1B | Tanh | ~96-97% |
| 1C | XOR | ~100% |
| 2A | Adam | ~97-98% |
| 3A | Basic CNN | ~70-75% |

---

## 📈 Visualizations Generated

1. **problem_1b_activation_comparison.png**
   - Training loss curves for ReLU, Sigmoid, Tanh
   - Test accuracy comparison

2. **problem_1c_xor_boundary.png**
   - 2D decision boundary for XOR problem
   - Shows non-linear separation

3. **problem_2a_optimizer_comparison.png**
   - Training loss for 4 optimizers
   - Test accuracy evolution

4. **problem_2b_gradient_analysis.png**
   - Log-scale gradient magnitudes
   - Demonstrates vanishing gradient problem

5. **problem_2c_regularization.png**
   - Train vs test accuracy
   - Comparison of regularization techniques

6. **problem_3a_cnn_training.png**
   - Training/validation loss curves
   - Validation accuracy over epochs

7. **problem_3a_confusion_matrix.png**
   - 10×10 heatmap for CIFAR-10 classes
   - Shows classification patterns

8. **problem_3c_filters.png**
   - 8 learned convolutional filters
   - Visualizes edge and pattern detectors

9. **problem_3c_activations.png**
   - Activation maps for 3 sample images
   - Shows feature detection in action

---

## 🔍 Code Structure

### Main Classes

#### `FlexibleMLP`
```python
FlexibleMLP(input_size, hidden_sizes, output_size, activation='relu')
```
- Supports arbitrary architecture
- Multiple activation functions
- Used in Problems 1 and 2

#### `XORMLP`
```python
XORMLP(hidden_size=4)
```
- Specialized for XOR problem
- Binary classification with sigmoid output

#### `DeepMLP`
```python
DeepMLP(activation='sigmoid')
```
- 6-layer deep network
- Used for gradient analysis

#### `MLPWithDropout`
```python
MLPWithDropout(dropout_p=0.5)
```
- Includes dropout layers
- Used for regularization comparison

#### `BasicCNN`
```python
BasicCNN()
```
- 2 convolutional layers
- 2 fully connected layers
- Used for CIFAR-10 classification

### Key Functions

#### `train_model()`
```python
train_model(model, train_loader, test_loader, optimizer, epochs, track_gradients)
```
- Generic training loop
- Optional gradient tracking
- Returns losses, accuracies, and gradients

#### `evaluate_model()`
```python
evaluate_model(model, test_loader)
```
- Computes test accuracy
- No gradient computation

---

## 🛠️ Customization

### Changing Hyperparameters

Edit these sections in the main script:

```python
# Learning rates
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Change lr here

# Batch sizes
train_loader = DataLoader(dataset, batch_size=64)  # Change batch_size

# Number of epochs
train_model(model, train_loader, test_loader, optimizer, epochs=20)  # Change epochs
```

### Adding New Architectures

```python
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define your layers here
        
    def forward(self, x):
        # Define forward pass
        return x
```

---

## 📝 Report Guidelines

Your PDF report should include:

### For Each Problem:
1. **Code snippets** (key functions only)
2. **Results tables** with accuracies and metrics
3. **Visualizations** (embedded plots)
4. **Analysis** (as specified in assignment)

### Analysis Requirements:

**Problem 1B**: 3-4 sentences on activation function performance

**Problem 2A**: 4-5 sentences on optimizer comparison

**Problem 2B**: 3-4 sentences on vanishing gradient observations

**Problem 2C**: 2-3 sentences on regularization effectiveness

**Problem 3B**: 4-5 sentences on architecture trade-offs (if completed)

---

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size in DataLoader
train_loader = DataLoader(dataset, batch_size=32)  # Instead of 64
```

#### Slow Training (CPU)
```python
# Reduce epochs or use smaller models for testing
train_model(model, train_loader, test_loader, optimizer, epochs=5)
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### Dataset Download Fails
```python
# Manually set download=True and check internet connection
torchvision.datasets.MNIST(root='./data', train=True, download=True)
```

---

## 📚 Additional Resources

### PyTorch Documentation
- [nn.Module Guide](https://pytorch.org/docs/stable/nn.html)
- [Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### Deep Learning Concepts
- [Understanding Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Optimization Algorithms](https://pytorch.org/docs/stable/optim.html)
- [CNN Architectures](https://pytorch.org/vision/stable/models.html)

---

## 🎓 Submission Checklist

- [ ] `dl_hw1_solution.py` (or `.ipynb`)
- [ ] `requirements.txt`
- [ ] PDF report with:
  - [ ] All visualizations embedded
  - [ ] Results tables
  - [ ] Analysis sections (as required)
  - [ ] Code snippets (key implementations)
- [ ] All 9 output plots generated
- [ ] Code runs without errors
- [ ] Results are reproducible

---

## ⚠️ Academic Integrity

This solution is provided as a reference and learning tool. Please:
- Understand each component before submission
- Modify and experiment with the code
- Write your own analysis and insights
- Follow your institution's academic integrity policies

---

## 📧 Questions?

If you encounter issues:
1. Check the Troubleshooting section
2. Review PyTorch documentation
3. Verify all dependencies are installed
4. Ensure datasets downloaded correctly

---

## 📄 License

This educational material is provided for learning purposes.

---

**Last Updated**: December 31, 2025  
**Course**: Deep Learning  
**Assignment**: Homework 1  
**Total Points**: 25