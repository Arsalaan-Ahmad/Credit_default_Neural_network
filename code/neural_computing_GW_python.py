import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import copy
import pandas as pd

# Debugging configuration
DEBUG = True
PRINT_EVERY = 50  # Print progress every N epochs

class RobustMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Enhanced initialization with debugging
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        if DEBUG:
            print(f"Initialized MLP with input_size={input_size}, hidden={hidden_size}")

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        return torch.sigmoid(self.fc2(x)).squeeze()

def safe_train(X_train, y_train, X_val, y_val, params):
    """Training with enhanced safeguards"""
    device = torch.device('cpu')
    
    try:
        # Model initialization
        model = RobustMLP(X_train.shape[1], params['hidden_size'])
        model.to(device)
        best_weights = copy.deepcopy(model.state_dict())
        best_f1 = -np.inf
        
        # Debugging info
        if DEBUG:
            print(f"\nStarting training with params: {params}")
            print(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Class balance - Train: {np.mean(y_train):.2f}, Val: {np.mean(y_val):.2f}")

        # Training setup
        optimizer = optim.SGD(model.parameters(), 
                            lr=params['lr'],
                            weight_decay=params['lambda'])
        
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)

        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_tensor)
            loss = nn.BCELoss()(outputs, y_tensor)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Validation
            if epoch % PRINT_EVERY == 0 or epoch == params['epochs']-1:
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_preds = (val_outputs >= 0.5).float().numpy()
                    current_f1 = f1_score(y_val, val_preds, zero_division=0)
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_weights = copy.deepcopy(model.state_dict())
                        if DEBUG:
                            print(f"Epoch {epoch}: New best F1 {current_f1:.4f}")

        # Final validation check
        if not isinstance(best_weights, dict):
            raise ValueError("Invalid state_dict detected")

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        if 'model' in locals():
            best_weights = copy.deepcopy(model.state_dict())
            best_f1 = -1
        else:
            # Fallback initialization
            model = RobustMLP(X_train.shape[1], params['hidden_size'])
            best_weights = copy.deepcopy(model.state_dict())
            best_f1 = -1

    return best_weights, best_f1

def stable_grid_search(X, y, param_grid):
    """Sequential grid search with detailed logging"""
    best_score = -np.inf
    best_params = None
    best_state = None
    
    print("\nStarting grid search...")
    print(f"Parameter grid: {param_grid}")
    print(f"Data shape: {X.shape}, positive samples: {np.mean(y):.2%}")
    
    for hidden_size in param_grid['hidden_size']:
        for lr in param_grid['lr']:
            for lam in param_grid['lambda']:
                current_params = {
                    'hidden_size': hidden_size,
                    'lr': lr,
                    'lambda': lam,
                    'epochs': 1000
                }
                
                print(f"\nTesting combination: {current_params}")
                
                try:
                    kf = KFold(n_splits=3)
                    fold_scores = []
                    
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                        print(f"\nFold {fold+1}")
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        weights, fold_f1 = safe_train(X_train, y_train, X_val, y_val, current_params)
                        fold_scores.append(fold_f1)
                        print(f"Fold {fold+1} F1: {fold_f1:.4f}")
                        
                        # Immediate validation check
                        test_model = RobustMLP(X.shape[1], hidden_size)
                        try:
                            test_model.load_state_dict(weights)
                        except Exception as e:
                            print(f"State_dict validation failed: {str(e)}")
                            fold_scores[-1] = 0.0
                    
                    avg_score = np.mean(fold_scores)
                    print(f"Average F1: {avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = current_params.copy()
                        best_state = copy.deepcopy(weights)
                        print(f"New best parameters: {best_params}")
                        
                except Exception as e:
                    print(f"Grid search iteration failed: {str(e)}")
    
    print("\nGrid search completed")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_state

if __name__ == '__main__':
    # Data loading with error handling
    try:
        data = pd.read_excel('C:\\Users\\arsal\\OneDrive\\Neural computing GW\\default of credit card clients.xls', header=1)
        print("Data loaded successfully")
        
        X = data.iloc[:, :-1].values.astype(np.float32)
        y = data.iloc[:, -1].values.astype(bool)
        
        # Data cleaning
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"Clean data shape: {X.shape}, positive ratio: {np.mean(y):.2%}")
        
        # Normalization
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        exit()

    # Parameter grid
    param_grid = {
        'hidden_size': [15,176],
        'lr': [0.95,0.2,0.01],
        'lambda': [0.001]
    }
    
    # Grid search
    best_params, best_state = stable_grid_search(X, y, param_grid)
    
    # Final model validation
    try:
        final_model = RobustMLP(X.shape[1], best_params['hidden_size'])
        final_model.load_state_dict(best_state)
        print("\nFinal model loaded successfully")
        
        with torch.no_grad():
            outputs = final_model(torch.FloatTensor(X))
            preds = (outputs >= 0.5).float().numpy()
            print(f"Final F1: {f1_score(y, preds):.4f}")
            print(f"Accuracy: {np.mean(preds == y):.4f}")
            
    except Exception as e:
        print(f"Final validation failed: {str(e)}")