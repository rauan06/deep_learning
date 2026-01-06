"""
Deep Learning Homework 1 - Complete Solution
Covers: MLPs, Optimization, and CNNs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

# Fast mode trades some accuracy for much faster runtime (smaller datasets, fewer epochs)
FAST_MODE = True

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} | FAST_MODE={FAST_MODE}")

# ============================================================================
# PROBLEM 1: IMPLEMENTING AND TRAINING MLPs
# ============================================================================

class FlexibleMLP(nn.Module):
    """Flexible MLP with configurable layers and activations"""
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(FlexibleMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_name = activation
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on output layer
        return x

# Load MNIST dataset
def load_mnist(fast: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # In fast mode, use smaller subsets to speed up experiments
    if fast:
        from torch.utils.data import Subset
        train_subset_size = 10000
        test_subset_size = 2000
        train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, test_loader, optimizer, epochs=10, 
                track_gradients=False):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []
    gradient_stats = [] if track_gradients else None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Track gradients if needed
            if track_gradients:
                grads = []
                for name, param in model.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        grads.append(param.grad.abs().mean().item())
                gradient_stats.append(grads)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return train_losses, test_accuracies, gradient_stats

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# ============================================================================
# PROBLEM 1A: MLP Implementation
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1A: MLP Implementation")
print("="*80)

train_loader, test_loader = load_mnist(fast=FAST_MODE)

model_1a = FlexibleMLP(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10,
    activation='relu'
).to(device)

optimizer_1a = optim.Adam(model_1a.parameters(), lr=0.001)
epochs_1a = 3 if FAST_MODE else 10
losses_1a, accs_1a, _ = train_model(model_1a, train_loader, test_loader, optimizer_1a, epochs=epochs_1a)

print(f"\nFinal Test Accuracy: {accs_1a[-1]:.4f}")

# ============================================================================
# PROBLEM 1B: Activation Function Comparison
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1B: Activation Function Comparison")
print("="*80)

activations = ['relu', 'sigmoid', 'tanh']
results_1b = {}

for act in activations:
    print(f"\nTraining with {act.upper()} activation...")
    model = FlexibleMLP(784, [128, 64], 10, activation=act).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs_1b = 3 if FAST_MODE else 10
    losses, accs, _ = train_model(model, train_loader, test_loader, optimizer, epochs=epochs_1b)
    results_1b[act] = {'losses': losses, 'accuracies': accs}

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for act in activations:
    plt.plot(results_1b[act]['losses'], label=act.upper())
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for act in activations:
    plt.plot(results_1b[act]['accuracies'], label=act.upper())
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('problem_1b_activation_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'problem_1b_activation_comparison.png'")

print("\n--- Activation Function Results ---")
for act in activations:
    print(f"{act.upper()}: Final Test Accuracy = {results_1b[act]['accuracies'][-1]:.4f}")

# ============================================================================
# PROBLEM 1C: Solving XOR
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1C: Solving XOR")
print("="*80)

# XOR dataset
X_xor = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORMLP(nn.Module):
    def __init__(self, hidden_size=4):
        super(XORMLP, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

xor_model = XORMLP(hidden_size=4).to(device)
xor_optimizer = optim.Adam(xor_model.parameters(), lr=0.1)
criterion = nn.BCELoss()

# Train XOR model
X_xor, y_xor = X_xor.to(device), y_xor.to(device)
xor_epochs = 500 if FAST_MODE else 2000
for epoch in range(xor_epochs):
    xor_model.train()
    xor_optimizer.zero_grad()
    outputs = xor_model(X_xor)
    loss = criterion(outputs, y_xor)
    loss.backward()
    xor_optimizer.step()
    
    if (epoch + 1) % max(100, xor_epochs // 4) == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Test XOR
xor_model.eval()
with torch.no_grad():
    predictions = xor_model(X_xor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_xor).float().mean()
    print(f"\nXOR Accuracy: {accuracy:.4f}")
    print("\nPredictions:")
    for i, (inp, pred, true) in enumerate(zip(X_xor, predictions, y_xor)):
        print(f"Input: {inp.cpu().numpy()}, Predicted: {pred.item():.4f}, True: {true.item()}")

# Visualize decision boundary
def plot_xor_decision_boundary(model):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        Z = model(grid).cpu().numpy()
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Output')
    
    # Plot XOR points
    X_np = X_xor.cpu().numpy()
    y_np = y_xor.cpu().numpy()
    plt.scatter(X_np[y_np.flatten() == 0, 0], X_np[y_np.flatten() == 0, 1], 
                c='blue', s=200, edgecolors='black', label='Class 0', marker='o')
    plt.scatter(X_np[y_np.flatten() == 1, 0], X_np[y_np.flatten() == 1, 1], 
                c='red', s=200, edgecolors='black', label='Class 1', marker='s')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('XOR Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('problem_1c_xor_boundary.png', dpi=300, bbox_inches='tight')
    print("\nDecision boundary plot saved as 'problem_1c_xor_boundary.png'")

plot_xor_decision_boundary(xor_model)

# ============================================================================
# PROBLEM 2A: Optimizer Comparison
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 2A: Optimizer Comparison")
print("="*80)

optimizers_config = {
    'SGD': {'lr': 0.01},
    'SGD_Momentum': {'lr': 0.01, 'momentum': 0.9},
    'RMSprop': {'lr': 0.001},
    'Adam': {'lr': 0.001}
}

results_2a = {}

for opt_name, config in optimizers_config.items():
    print(f"\nTraining with {opt_name}...")
    model = FlexibleMLP(784, [128, 64], 10, activation='relu').to(device)
    
    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif opt_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    start_time = time.time()
    epochs_2a = 5 if FAST_MODE else 20
    losses, accs, _ = train_model(model, train_loader, test_loader, optimizer, epochs=epochs_2a)
    training_time = time.time() - start_time
    
    results_2a[opt_name] = {
        'losses': losses,
        'accuracies': accs,
        'time': training_time
    }

# Plot optimizer comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for opt_name in optimizers_config.keys():
    ax1.plot(results_2a[opt_name]['losses'], label=opt_name)
    ax2.plot(results_2a[opt_name]['accuracies'], label=opt_name)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss by Optimizer')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Test Accuracy by Optimizer')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('problem_2a_optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'problem_2a_optimizer_comparison.png'")

print("\n--- Optimizer Results ---")
for opt_name in optimizers_config.keys():
    final_acc = results_2a[opt_name]['accuracies'][-1]
    time_taken = results_2a[opt_name]['time']
    print(f"{opt_name}: Accuracy = {final_acc:.4f}, Time = {time_taken:.2f}s")

# ============================================================================
# PROBLEM 2B: Gradient Analysis
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 2B: Gradient Analysis")
print("="*80)

class DeepMLP(nn.Module):
    def __init__(self, activation='sigmoid'):
        super(DeepMLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 10)
        ])
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        self.activation_name = activation
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

gradient_results = {}

for act in ['sigmoid', 'relu']:
    print(f"\nAnalyzing gradients with {act.upper()}...")
    model = DeepMLP(activation=act).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs_2b = 2 if FAST_MODE else 5
    _, _, grad_stats = train_model(model, train_loader, test_loader, optimizer, 
                                    epochs=epochs_2b, track_gradients=True)
    gradient_results[act] = np.array(grad_stats)

# Plot gradient magnitudes
plt.figure(figsize=(12, 6))
for act in ['sigmoid', 'relu']:
    grads = gradient_results[act]
    mean_grads = grads.mean(axis=0)
    plt.plot(range(1, len(mean_grads) + 1), mean_grads, marker='o', label=act.upper())

plt.xlabel('Layer Depth')
plt.ylabel('Mean Gradient Magnitude')
plt.title('Gradient Magnitudes Across Layers')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig('problem_2b_gradient_analysis.png', dpi=300, bbox_inches='tight')
print("\nGradient analysis plot saved as 'problem_2b_gradient_analysis.png'")

# ============================================================================
# PROBLEM 2C: Regularization Effects
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 2C: Regularization Effects")
print("="*80)

class MLPWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(MLPWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

reg_configs = {
    'No Regularization': {'weight_decay': 0, 'dropout': False},
    'L2 (weight_decay=0.001)': {'weight_decay': 0.001, 'dropout': False},
    'Dropout (p=0.5)': {'weight_decay': 0, 'dropout': True}
}

reg_results = {}

for config_name, config in reg_configs.items():
    print(f"\nTraining with {config_name}...")
    
    if config['dropout']:
        model = MLPWithDropout(dropout_p=0.5).to(device)
    else:
        model = FlexibleMLP(784, [128, 64], 10, activation='relu').to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=config['weight_decay'])
    
    # Track both train and test accuracy
    criterion = nn.CrossEntropyLoss()
    train_accs = []
    test_accs = []
    
    epochs_2c = 5 if FAST_MODE else 20
    for epoch in range(epochs_2c):
        model.train()
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_acc = correct_train / total_train
        test_acc = evaluate_model(model, test_loader)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    reg_results[config_name] = {'train': train_accs, 'test': test_accs}
    print(f"Final - Train Acc: {train_accs[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}")

# Plot regularization comparison
plt.figure(figsize=(14, 5))

for i, config_name in enumerate(reg_configs.keys()):
    plt.subplot(1, 3, i+1)
    plt.plot(reg_results[config_name]['train'], label='Train', linewidth=2)
    plt.plot(reg_results[config_name]['test'], label='Test', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(config_name)
    plt.legend()
    plt.grid(True)
    plt.ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig('problem_2c_regularization.png', dpi=300, bbox_inches='tight')
print("\nRegularization comparison saved as 'problem_2c_regularization.png'")

# ============================================================================
# PROBLEM 3A: Basic CNN Implementation
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3A: Basic CNN Implementation")
print("="*80)

# Load CIFAR-10
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_cifar
)
test_dataset_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_cifar
)

# In fast mode, subsample CIFAR-10 as well
if FAST_MODE:
    from torch.utils.data import Subset
    train_subset_size_cifar = 10000
    test_subset_size_cifar = 2000
    train_indices_cifar = torch.randperm(len(train_dataset_cifar))[:train_subset_size_cifar]
    test_indices_cifar = torch.randperm(len(test_dataset_cifar))[:test_subset_size_cifar]
    train_dataset_cifar = Subset(train_dataset_cifar, train_indices_cifar)
    test_dataset_cifar = Subset(test_dataset_cifar, test_indices_cifar)

train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=64, shuffle=True)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=1000, shuffle=False)

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = BasicCNN().to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train CNN
print("Training Basic CNN...")
cnn_train_losses = []
cnn_val_losses = []
cnn_val_accs = []
criterion = nn.CrossEntropyLoss()

epochs_3a = 5 if FAST_MODE else 20
for epoch in range(epochs_3a):
    cnn_model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader_cifar:
        inputs, labels = inputs.to(device), labels.to(device)
        cnn_optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        cnn_optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader_cifar)
    cnn_train_losses.append(avg_train_loss)
    
    # Validation
    cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader_cifar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(test_loader_cifar)
    val_acc = correct / total
    cnn_val_losses.append(avg_val_loss)
    cnn_val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/20, Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Plot CNN training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cnn_train_losses, label='Train Loss')
plt.plot(cnn_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cnn_val_accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('problem_3a_cnn_training.png', dpi=300, bbox_inches='tight')
print(f"\nFinal Test Accuracy: {cnn_val_accs[-1]:.4f}")
print("Training curves saved as 'problem_3a_cnn_training.png'")

# Confusion Matrix
cnn_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader_cifar:
        inputs = inputs.to(device)
        outputs = cnn_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CIFAR-10')
plt.savefig('problem_3a_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'problem_3a_confusion_matrix.png'")

# ============================================================================
# PROBLEM 3C: Visualizing Learned Features
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3C: Visualizing Learned Features")
print("="*80)

# Visualize first conv layer filters
weights = cnn_model.conv1.weight.data.cpu()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(8):
    filt = weights[i]
    filt = (filt - filt.min()) / (filt.max() - filt.min())
    filt = filt.permute(1, 2, 0).numpy()
    axes[i].imshow(filt)
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i+1}')

plt.suptitle('First Convolutional Layer Filters')
plt.tight_layout()
plt.savefig('problem_3c_filters.png', dpi=300, bbox_inches='tight')
print("Filters visualization saved as 'problem_3c_filters.png'")

# Visualize activation maps
def get_activation(model, x):
    activations = []
    def hook_fn(m, i, o):
        activations.append(o)
    
    handle = model.conv1.register_forward_hook(hook_fn)
    model(x)
    handle.remove()
    return activations[0]

# Get sample images
sample_images, _ = next(iter(test_loader_cifar))
sample_images = sample_images[:3].to(device)

cnn_model.eval()
with torch.no_grad():
    activations = get_activation(cnn_model, sample_images)

fig, axes = plt.subplots(3, 9, figsize=(18, 6))

for i in range(3):
    # Original image
    img = sample_images[i].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[i, 0].imshow(img)
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')
    
    # 8 activation maps
    for j in range(8):
        act_map = activations[i, j].cpu().numpy()
        axes[i, j+1].imshow(act_map, cmap='viridis')
        axes[i, j+1].set_title(f'Filter {j+1}')
        axes[i, j+1].axis('off')

plt.suptitle('Activation Maps from First Conv Layer')
plt.tight_layout()
plt.savefig('problem_3c_activations.png', dpi=300, bbox_inches='tight')
print("Activation maps saved as 'problem_3c_activations.png'")

print("\n" + "="*80)
print("ALL PROBLEMS COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("1. problem_1b_activation_comparison.png")
print("2. problem_1c_xor_boundary.png")
print("3. problem_2a_optimizer_comparison.png")
print("4. problem_2b_gradient_analysis.png")
print("5. problem_2c_regularization.png")
print("6. problem_3a_cnn_training.png")
print("7. problem_3a_confusion_matrix.png")
print("8. problem_3c_filters.png")
print("9. problem_3c_activations.png")
