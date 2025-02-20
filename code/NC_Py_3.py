import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if that works better for you
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data = pd.read_excel('default of credit card clients.xls')

# Convert all but the label column to numeric, coercing errors to NaN 
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Optionally, drop rows with NaN values (or handle them as needed)
data.dropna(inplace=True)

X = data.iloc[:, :-1].values.astype(np.float32)
y = data.iloc[:, -1].values.astype(bool)

# Class distribution
print(f'Class distribution: {100 * np.mean(y):.2f}% ones')

# Normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Class weights
num_class1 = np.sum(y)
num_class0 = np.sum(~y)
total_samples = num_class0 + num_class1
class_weights = np.array([total_samples / (2 * num_class0), total_samples / (2 * num_class1)], dtype=np.float32)

# Hyperparameters
hidden_sizes = [128, 144, 176]
learning_rates = [0.95, 0.9, 0.8]
lambda_value = 0.001
epochs = 1000
patience = 100  # Early stopping patience

# Initialize grid result matrices
grid_acc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_f1 = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_recall = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_train_time = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_predict_time = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_val_loss = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)
grid_train_loss = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32)

# K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1 = 0
best_params = {}

DEBUG = True

# MLP Training Function
def train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights, patience):
    input_size = X_train.shape[1]
    W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2 / input_size)
    B1 = np.zeros((hidden_size, 1), dtype=np.float32)
    W2 = np.random.randn(1, hidden_size).astype(np.float32) * np.sqrt(2 / hidden_size)
    B2 = np.zeros((1, 1), dtype=np.float32)

    weight_vector = class_weights[0] * (y_train == 0) + class_weights[1] * (y_train == 1)
    train_losses = []
    val_metrics = {'accuracy': [], 'f1': [], 'error': []}
    best_f1 = 0
    best_weights = (W1.copy(), B1.copy(), W2.copy(), B2.copy())
    wait = 0

    if DEBUG:
        print(f"Starting training with hidden_size: {hidden_size}, lr: {lr}, epochs: {epochs}")

    for epoch in range(epochs):
        # update learning rate schedule
        if epoch >= 0.6 * epochs:
            lr *= 0.5

        # Forward pass
        hidden_input = np.dot(W1, X_train.T) + B1
        hidden_output = np.maximum(0, hidden_input)
        output_input = np.dot(W2, hidden_output) + B2
        output_prob = 1 / (1 + np.exp(-output_input))

        # Loss calculation
        data_loss = -np.mean((y_train * np.log(output_prob + 1e-10) + (1 - y_train) * np.log(1 - output_prob + 1e-10)) * weight_vector)
        reg_loss = 0.5 * lambda_value * (np.sum(W1**2) + np.sum(W2**2))
        total_loss = data_loss + reg_loss
        train_losses.append(total_loss)

        # Backward pass
        dZ2 = (output_prob - y_train) * weight_vector / X_train.shape[0]
        dW2 = np.dot(dZ2, hidden_output.T)
        dB2 = np.sum(dZ2, axis=1, keepdims=True)
        dHidden = np.dot(W2.T, dZ2)
        dZ1 = dHidden * (hidden_output > 0)
        dW1 = np.dot(dZ1, X_train)
        dB1 = np.sum(dZ1, axis=1, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        B1 -= lr * dB1
        W2 -= lr * dW2
        B2 -= lr * dB2

        # Evaluate on validation set
        val_acc, val_f1, val_loss = evaluate_model(X_val, y_val, W1, B1, W2, B2)
        val_metrics['accuracy'].append(val_acc)
        val_metrics['f1'].append(val_f1)
        val_metrics['error'].append(val_loss)

        if DEBUG and epoch % 100 == 0:
            print(f"Epoch {epoch}: train_loss={total_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = (W1.copy(), B1.copy(), W2.copy(), B2.copy())
            wait = 0
            if DEBUG:
                print(f"Epoch {epoch}: New best f1 = {best_f1:.4f}")
        else:
            wait += 1
            if epoch > 650 and wait >= patience:
                if DEBUG:
                    print(f"Stopping early at epoch {epoch} due to no improvement.")
                break

    return best_weights, train_losses, val_metrics

# Model Evaluation Function
def evaluate_model(X, y, W1, B1, W2, B2):
    hidden_input = np.dot(W1, X.T) + B1
    hidden_output = np.maximum(0, hidden_input)
    output_input = np.dot(W2, hidden_output) + B2
    output_prob = 1 / (1 + np.exp(-output_input))
    predictions = (output_prob > 0.5).astype(int)
    accuracy = accuracy_score(y, predictions.flatten())
    f1 = f1_score(y, predictions.flatten())
    recall = recall_score(y, predictions.flatten())
    loss = -np.mean(y * np.log(output_prob + 1e-10) + (1 - y) * np.log(1 - output_prob + 1e-10))
    return accuracy, f1, loss

# Grid Search Function
def grid_search_fold(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights, patience):
    if DEBUG:
        print(f"Grid search fold: hidden_size={hidden_size}, lr={lr}")
    start_time = time()
    best_weights, train_losses, val_metrics = train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights, patience)
    train_time = time() - start_time
    if DEBUG:
        print(f"Training time: {train_time:.2f}s")

    # Measure prediction time on validation set
    start_pred = time()
    val_acc, val_f1, val_loss = evaluate_model(X_val, y_val, *best_weights)
    pred_time = time() - start_pred
    if DEBUG:
        print(f"Prediction time: {pred_time:.4f}s")
    
    # Calculate recall using the same prediction formula
    hidden_input = np.dot(best_weights[0], X_val.T) + best_weights[1]
    hidden_output = np.maximum(0, hidden_input)
    output_input = np.dot(best_weights[2], hidden_output) + best_weights[3]
    output_prob = 1 / (1 + np.exp(-output_input))
    predictions = (output_prob > 0.5).astype(int)
    val_recall = recall_score(y_val, predictions.flatten())
    
    return {
        'f1': val_f1,
        'acc': val_acc,
        'recall': val_recall,
        'train_time': train_time,
        'pred_time': pred_time,
        'val_loss': val_loss,
        'train_loss': train_losses[-1]
    }

# Parallel Grid Search
results = Parallel(n_jobs=2)(delayed(grid_search_fold)(
    X[train_idx], y[train_idx], X[val_idx], y[val_idx], hidden_size, lr, epochs, lambda_value, class_weights, patience
) for hidden_size in hidden_sizes for lr in learning_rates for train_idx, val_idx in cv.split(X))

# Aggregate results
for hs_idx, hidden_size in enumerate(hidden_sizes):
    for lr_idx, lr in enumerate(learning_rates):
        fold_metrics = {'f1': [], 'acc': [], 'recall': [], 'time': [], 'val_loss': [], 'train_loss': []}
        for fold in range(5):
            result = results[hs_idx * len(learning_rates) * 5 + lr_idx * 5 + fold]
            fold_metrics['f1'].append(result['f1'])
            fold_metrics['acc'].append(result['acc'])
            fold_metrics['recall'].append(result['recall'])
            fold_metrics['time'].append(result['time'])
            fold_metrics['val_loss'].append(result['val_loss'])
            fold_metrics['train_loss'].append(result['train_loss'])

        # Store average metrics
        grid_acc[hs_idx, lr_idx] = np.mean(fold_metrics['acc'])
        grid_f1[hs_idx, lr_idx] = np.mean(fold_metrics['f1'])
        grid_recall[hs_idx, lr_idx] = np.mean(fold_metrics['recall'])
        grid_train_time[hs_idx, lr_idx] = np.mean(fold_metrics['time'])
        grid_val_loss[hs_idx, lr_idx] = np.mean(fold_metrics['val_loss'])
        grid_train_loss[hs_idx, lr_idx] = np.mean(fold_metrics['train_loss'])

        # Update best parameters
        if grid_f1[hs_idx, lr_idx] > best_f1:
            best_f1 = grid_f1[hs_idx, lr_idx]
            best_params = {'hidden_size': hidden_size, 'learning_rate': lr}

# Heatmaps
plt.ion()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(grid_acc, annot=True, xticklabels=learning_rates, yticklabels=hidden_sizes, cmap='viridis')
plt.title('Accuracy Heatmap')
plt.xlabel('Learning Rate')
plt.ylabel('Hidden Size')

plt.subplot(1, 2, 2)
sns.heatmap(grid_train_time, annot=True, xticklabels=learning_rates, yticklabels=hidden_sizes, cmap='hot')
plt.title('Training Time Heatmap')
plt.xlabel('Learning Rate')
plt.ylabel('Hidden Size')
plt.show(block=True)

# Final Model Training
print(f'\nTraining final model with hidden size: {best_params["hidden_size"]}, learning rate: {best_params["learning_rate"]}')
final_weights, train_losses, val_metrics = train_mlp(X, y, X, y, best_params['hidden_size'], best_params['learning_rate'], epochs, lambda_value, class_weights, patience)

# Final Evaluation
final_acc, final_f1, final_loss = evaluate_model(X, y, *final_weights)
print(f'\nFinal Performance:\n Accuracy: {final_acc:.4f}\n F1: {final_f1:.4f}\n Loss: {final_loss:.4f}')

# Final Evaluation & Plotting

# Compute output probabilities from the final model using the final weights.
hidden_input = np.dot(final_weights[0], X.T) + final_weights[1]
hidden_output = np.maximum(0, hidden_input)
output_input = np.dot(final_weights[2], hidden_output) + final_weights[3]
output_prob = 1 / (1 + np.exp(-output_input))

# Ensure ground truth labels are integers (in case they are Boolean)
y_true = y.astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, output_prob.flatten())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
y_pred = (output_prob > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred.flatten())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

input("Press Enter to exit...")

