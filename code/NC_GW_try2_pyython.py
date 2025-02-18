import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve)

def load_data():
    # Load and preprocess data
    data = pd.read_excel(r'C:\Users\arsal\OneDrive\Neural computing GW\default of credit card clients.xls', header=1)
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(bool)
    
    # Clean data
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Class weights
    num_class0 = np.sum(~y)
    num_class1 = np.sum(y)
    total = len(y)
    class_weights = np.array([total/(2*num_class0), total/(2*num_class1)], dtype=np.float32)
    
    print(f'Class distribution: {np.mean(y)*100:.2f}% positives')
    return X, y, class_weights

def initialize_parameters(input_size, hidden_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2/input_size)
    B1 = np.zeros((hidden_size, 1), dtype=np.float32)  # (hidden_size, 1)
    W2 = np.random.randn(1, hidden_size).astype(np.float32) * np.sqrt(2/hidden_size)
    B2 = np.zeros((1, 1), dtype=np.float32)  # Changed to (1, 1)
    return W1, B1, W2, B2

def forward_pass(X, W1, B1, W2, B2):
    hidden_input = W1 @ X.T + B1  # B1 shape (hidden_size, 1)
    hidden_output = np.maximum(0, hidden_input)
    output = W2 @ hidden_output + B2  # B2 shape (1, 1)
    prob = 1 / (1 + np.exp(-output))
    return hidden_output, prob

def backward_pass(X, y, weights, hidden_output, prob, W1, B1, W2, B2, lambda_val):
    m = X.shape[0]
    batch_size = 512
    epsilon = 1e-8

    # Initialize gradients with proper dimensions
    dW1 = np.zeros_like(W1)
    dB1 = np.zeros_like(B1)  # (hidden_size, 1)
    dW2 = np.zeros_like(W2)
    dB2 = np.zeros_like(B2)  # (1, 1)

    for i in range(0, m, batch_size):
        end_idx = min(i+batch_size, m)
        X_batch = X[i:end_idx].T
        y_batch = y[i:end_idx]
        w_batch = weights[i:end_idx]
        h_batch = hidden_output[:, i:end_idx]
        p_batch = prob[:, i:end_idx]

         # Output layer gradients
        dZ2 = (p_batch - y_batch) * w_batch / (end_idx - i)
        dW2 += dZ2 @ h_batch.T
        dB2 += np.sum(dZ2, axis=1, keepdims=True)  # Now matches (1, 1) shape
        
        # Hidden layer gradients
        dHidden = W2.T @ dZ2
        dZ1 = dHidden * (h_batch > 0).astype(np.float32)
        dW1 += dZ1 @ X_batch.T
        dB1 += np.sum(dZ1, axis=1, keepdims=True)  # Maintains (hidden_size, 1) shape

    # ... [regularization code] ...
    return dW1, dB1, dW2, dB2

def evaluate_model(X, y, W1, B1, W2, B2):
    batch_size = 1024
    prob = np.zeros(len(y), dtype=np.float32)
    pred = np.zeros(len(y), dtype=bool)
    
    for i in range(0, len(y), batch_size):
        end_idx = min(i+batch_size, len(y))
        _, p = forward_pass(X[i:end_idx], W1, B1, W2, B2)
        prob[i:end_idx] = p.squeeze()
        pred[i:end_idx] = p.squeeze() > 0.5

    # Calculate metrics with numerical stability
    try:
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        auc = roc_auc_score(y, prob)
    except:
        acc = 0.0
        f1 = 0.0
        auc = 0.0
    
    return acc, f1, pred, prob

def train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, lambda_val, epochs, class_weights):
    W1, B1, W2, B2 = initialize_parameters(X_train.shape[1], hidden_size)
    weights = np.where(y_train, class_weights[1], class_weights[0])
    
    train_loss = []
    val_metrics = {'F1': [], 'Accuracy': [], 'Error': [], 'TrainAccuracy': []}
    best_f1 = 0
    best_weights = (W1, B1, W2, B2)
    patience = 100
    wait = 0
    
    for epoch in range(epochs):
        # Forward pass
        hidden_output, prob = forward_pass(X_train, W1, B1, W2, B2)
        
        # Loss calculation with numerical stability
        epsilon = 1e-8
        loss = -np.mean(weights * (y_train * np.log(prob + epsilon) + 
                                  (1 - y_train) * np.log(1 - prob + epsilon)))
        reg_loss = 0.5 * lambda_val * (np.sum(W1**2) + np.sum(W2**2))
        total_loss = loss + reg_loss
        train_loss.append(total_loss)
        
        # Backward pass
        dW1, dB1, dW2, dB2 = backward_pass(X_train, y_train, weights, 
                                  hidden_output, prob, 
                                  W1, B1, W2, B2, lambda_val)
        
        # Learning rate schedule
        if epoch >= 0.6 * epochs:
            lr *= 0.5
        
        # Update parameters with gradient clipping
        max_grad_norm = 1.0
        grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2))
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / grad_norm
            dW1 *= scale
            dW2 *= scale
        
        W1 -= lr * dW1
        B1 -= lr * dB1
        W2 -= lr * dW2
        B2 -= lr * dB2
        
        # Validation
        train_acc, _, _, _ = evaluate_model(X_train, y_train, W1, B1, W2, B2)
        val_acc, val_f1, _, val_prob = evaluate_model(X_val, y_val, W1, B1, W2, B2)
        val_loss = -np.mean(y_val * np.log(val_prob + epsilon) + 
                           (1 - y_val) * np.log(1 - val_prob + epsilon))
        
        val_metrics['F1'].append(val_f1)
        val_metrics['Accuracy'].append(val_acc)
        val_metrics['Error'].append(val_loss)
        val_metrics['TrainAccuracy'].append(train_acc)
        
        # Early stopping
        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_weights = (W1.copy(), B1.copy(), W2.copy(), B2.copy())
            wait = 0
        else:
            wait += 1
            if epoch > 420 and wait >= patience:
                break
    
    return {
        'weights': best_weights,
        'train_loss': train_loss,
        'val_metrics': val_metrics
    }

def main():
    X, y, class_weights = load_data()
    
    # Hyperparameters
    hidden_sizes = [ 176]
    learning_rates = [0.95]
    lambda_val = 0.001
    epochs = 1000
    
    best_f1 = 0
    best_result = None
    
    # Manual grid search without parallel processing
    for hs in hidden_sizes:
        for lr in learning_rates:
            print(f'\nTesting hidden size: {hs}, lr: {lr:.2f}')
            
            kf = KFold(n_splits=5)
            fold_f1 = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f'  Fold {fold+1}', end='')
                result = train_mlp(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                                  hs, lr, lambda_val, epochs, class_weights)
                max_f1 = max(result['val_metrics']['F1'])
                fold_f1.append(max_f1)
                print(f' - Max F1: {max_f1:.4f}')
            
            avg_f1 = np.mean(fold_f1)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_result = result
                print(f'New best average F1: {avg_f1:.4f}')
    
    # Final evaluation
    W1, B1, W2, B2 = best_result['weights']
    acc, f1, pred, prob = evaluate_model(X, y, W1, B1, W2, B2)
    
    print(f'\nFinal Metrics:')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {roc_auc_score(y, prob):.4f}')
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # ROC Curve
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = roc_curve(y, prob)
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve (AUC = {roc_auc_score(y, prob):.3f})')
    
    # Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y, pred)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Training Loss
    plt.subplot(2, 2, 3)
    plt.plot(best_result['train_loss'])
    plt.title('Training Loss')
    
    # Validation F1
    plt.subplot(2, 2, 4)
    plt.plot(best_result['val_metrics']['F1'])
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()