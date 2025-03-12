import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight # Import class_weight function
import matplotlib
matplotlib.use('TkAgg') # Change backend to show plots on macOS (TkAgg) 
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
data = pd.read_excel('default of credit card clients.xls')

# Convert features to numeric using pandas
data.iloc[:, :-1] = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

X = data.iloc[:, :-1].values.astype(np.float32)#.astype(np.float32) we tried using float64 be it was performing the same as well so we used float32
y = data.iloc[:, -1].values.astype(bool) # Convert to boolean values

# Initial split (80/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalization using sklearn built-in (we are performing z-score normalization)
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# Calculate balanced class weights
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train_full), y=y_train_full
).astype(np.float32)

print(f'Class distribution (train): {100 * np.mean(y_train_full):.2f}% positive class')
print(f'Computed class weights: {class_weights}')

# Hyperparameters
hidden_sizes = [12,17,24,32,48]
lambda_value = 0.001  # L2 regularization
epochs = 500  
DEBUG = True
# here we used deepseek to find how to perform the LRRT from the Lesie Smith's 2017 paper, which was mentioned in the section 3.3 of the paper.
#prompt used for this: im trying to use the method mentioned in this paper "L. N. Smith, "Cyclical Learning Rates for Training Neural Networks," 2017 IEEE Winter Conference on Applications of Computer Vision (WACV), Santa Rosa, CA, USA, 2017, pp. 464-472, doi: 10.1109/WACV.2017.58. keywords: {Training;Neural networks;Schedules;Computer architecture;Tuning;Computational efficiency}," section 3.3. i want to implement the learning rate range test (LRRT) to find the optimal learning rate for my neural network. can you help me with that?
# -------------------------
# Triangular LR Range Test (Modified with class weights)
# -------------------------
#this function returns the learning rate, accuracy and loss for each iteration. which is used to find the base_lr and max_lr. using the defined function find_lr_bounds.
def triangular_lr_test(X_train, y_train, hidden_size, min_lr, max_lr, num_iters, lambda_value, class_weights):

    input_size = X_train.shape[1] # Input size for the model (number of features)
    
    # He initialization with numpy
    he_scale = lambda fan_in: np.sqrt(2/fan_in) # He initialization scaling factor
    W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * he_scale(input_size) # He initialization for W1
    B1 = np.zeros((hidden_size, 1), dtype=np.float32) # Initialize bias to zeros
    W2 = np.random.randn(1, hidden_size).astype(np.float32) * he_scale(hidden_size) # He initialization for W2 
    B2 = np.zeros((1, 1), dtype=np.float32) # Initialize bias to zeros
    
    # Create sample weights vector using sklearn-computed weights
    weight_vector = np.where(y_train, class_weights[1], class_weights[0])
    
    lrs, accs, losses = [], [], []
    
    for i in range(num_iters):
        lr = np.interp(i, [0, num_iters], [min_lr, max_lr])
        lrs.append(lr)
        
        # Forward pass with numpy
        hidden = np.maximum(0, W1 @ X_train.T + B1) # ReLU activation for hidden layer
        output = 1/(1 + np.exp(-(W2 @ hidden + B2))) # Sigmoid activation for output layer
        output = np.clip(output, 1e-10, 1-1e-10) # Clip to prevent log(0)
        
        # Weighted loss calculation
        loss_per_sample = - (y_train * np.log(output + 1e-10) + (1 - y_train) * np.log(1 - output + 1e-10)) # Cross-entropy loss per sample 
        weighted_loss = np.mean(loss_per_sample * weight_vector) # Weighted loss function 
        reg_loss = 0.5 * lambda_value * (np.sum(W1**2) + np.sum(W2**2)) # L2 regularization
        
        # Store metrics
        accs.append(accuracy_score(y_train, (output > 0.5).astype(int).ravel()))
        losses.append(weighted_loss + reg_loss)
        
        # Backprop with numpy
        dZ2 = (output - y_train) * weight_vector / X_train.shape[0] # Weighted loss gradient
        W2_grad = dZ2 @ hidden.T + lambda_value * W2 # Add regularization term gradient
        B2_grad = dZ2.sum(axis=1, keepdims=True) # Keepdims to ensure correct shape for broadcasting in numpy
        dHidden = (W2.T @ dZ2) * (hidden > 0) # ReLU gradient for hidden layer
        W1_grad = dHidden @ X_train + lambda_value * W1 # Add regularization term gradient
        B1_grad = dHidden.sum(axis=1, keepdims=True) # Keepdims to ensure correct shape for broadcasting in numpy
        
        # Parameter updates
        W2 -= lr * W2_grad # Update weights and biases with gradients (gradient descent)
        B2 -= lr * B2_grad
        W1 -= lr * W1_grad
        B1 -= lr * B1_grad
    
    return np.array(lrs), np.array(accs), np.array(losses)

# Function to find base_lr and max_lr
#this function returns the base_lr and max_lr. which are used to find the learning rates for the grid search.
def find_lr_bounds(lrs, accs, losses):
    #Identifies base_lr and max_lr per Smith's paper
    # Find base_lr: First LR where accuracy exceeds 10% of max accuracy
    threshold_acc = 0.1 * np.max(accs) # 10% of max accuracy threshold
    rising_mask = accs > threshold_acc
    if np.any(rising_mask):
        base_lr = lrs[np.argmax(rising_mask)]
    else:
        base_lr = lrs[0]
        
    # Find max_lr: First LR where loss exceeds 2x minimum loss
    min_loss = np.min(losses)
    divergence_mask = losses > 2 * min_loss
    if np.any(divergence_mask):
        max_lr = lrs[np.argmax(divergence_mask)]
    else:
        max_lr = lrs[-1]
    
    return base_lr, max_lr

# -------------------------
# Execute LRRT with Class Weights
# -------------------------
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train_full), y=y_train_full
).astype(np.float32) # Recompute class weights for full training set 

# Create a representative subset of the training data for LRRT
subset_size = 1000  # Using 1K samples for quick testing
subset_idx = np.random.choice(len(X_train_full), subset_size, replace=False) # Randomly sample subset_size indices without replacement
X_subset = X_train_full[subset_idx]
y_subset = y_train_full[subset_idx]

# Runing the test with class weights
lrs, accs, losses = triangular_lr_test(
    X_subset, y_subset, 
    hidden_size=24, 
    min_lr=1e-3, # Started with a small LR was 1e-5 before
    max_lr=1, 
    num_iters=100, # 100 iterations
    lambda_value=0.001,
    class_weights=class_weights  # Passing class weights here
)

# Plot results
# Plotting the learning rate, accuracy and loss for each iteration.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.semilogx(lrs, accs) # Use semilogx for log scale on x-axis 
ax1.set_title("Accuracy vs Learning Rate")
ax1.set_ylabel("Accuracy")

ax2.semilogx(lrs, losses) # Use semilogx for log scale on x-axis
ax2.set_title("Loss vs Learning Rate")
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Loss")

plt.tight_layout()
plt.show()

# Calculate bounds
base_lr, max_lr = find_lr_bounds(lrs, accs, losses)
print(f"Recommended base_lr: {base_lr:.5f}")
print(f"Recommended max_lr: {max_lr:.5f}")

learning_rates = np.geomspace(base_lr, max_lr, num=5) # Geometrically spaced learning rates
print(learning_rates)
# -------------------------

# Initialize grid result matrices: accuracy, training time, and AUC
grid_acc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) # Initialize with zeros
grid_train_time = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) # Initialize with zeros
grid_auc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) # Initialize with zeros
best_auc = 0
best_params = {}

# MLP Training Function. This function trains the mlp model and performs backpropagation and returns weight and bias.
# stochastic gradient descent (SGD) is applied here.
def train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights):
    #  section 1:
    # here,
    # w1: weight for input to hidden layer. Initialized using He initialization
    # b1: bias for hidden layer 
    # w2: weight from hidden to output layer. Same initialization as w1.
    # b2: bias for output layer
    # weight_vector is calculated to implement weight loss function which gives different importance to class based on class weight which we take in as input. (is helps deal with the imbalence dataset)
    input_size = X_train.shape[1]
    W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2 / input_size) # He initialization for W1
    B1 = np.zeros((hidden_size, 1), dtype=np.float32) # Initialize bias to zeros 
    W2 = np.random.randn(1, hidden_size).astype(np.float32) * np.sqrt(2 / hidden_size) # He initialization for W2 
    B2 = np.zeros((1, 1), dtype=np.float32) # Initialize bias to zeros
    weight_vector = class_weights[0]*(y_train == 0) + class_weights[1]*(y_train == 1) # Create weight vector for weighted loss function 

    if DEBUG:
        print(f"Starting training with hidden_size: {hidden_size}, lr: {lr}, epochs: {epochs}")
    #section2:
    #here,
    #for each epoc the network undergoes the following steps:

    for epoch in range(epochs):
        #2.1 forward propagation
        # compute input to the hiden layer 'hidden_input'
        # hiden layer: applying ReLU activation
        # compute input to the output layer 'output_input'
        # output layer: applying sigmoid function and clipping to prevent log(0)
        
        hidden_input = np.dot(W1, X_train.T) + B1 # Linear input layer to hidden layer
        hidden_output = np.maximum(0, hidden_input) # ReLU activation for hidden layer
        output_input = np.dot(W2, hidden_output) + B2 # Linear output layer from hidden layer
        output_prob = 1 / (1 + np.exp(-output_input))# Sigmoid activation
        output_prob = np.clip(output_prob, 1e-10, 1-1e-10) # Clip to prevent log(0)
        data_loss = -np.mean((y_train * np.log(output_prob + 1e-10) +
                              (1 - y_train) * np.log(1 - output_prob + 1e-10)) * weight_vector) # Weighted loss function
        
        
        
        #2.2 loss calculation
        # we are using weighted binary cross-entropy loss function as we are performing binary classification
        #L2 regularization term is added to prevent overfitting
        reg_loss = 0.5 * lambda_value * (np.sum(W1**2) + np.sum(W2**2)) # L2 regularization
        total_loss = data_loss + reg_loss # Regularized loss function
        
        
        # section 3: backpropagation
        dZ2 = (output_prob - y_train) * weight_vector / X_train.shape[0] # calculating  gradient of loss w.r.t. output layer
        dW2 = np.dot(dZ2, hidden_output.T) + lambda_value * W2 # calculating gradient w.r.t. W2 (including L2 regularization)
        dB2 = np.sum(dZ2, axis=1, keepdims=True) # Gradient w.r.t. B2 # Keepdims to ensure correct shape for broadcasting in numpy by summing along axis 1. which means summing along columns.
        
        dHidden = np.dot(W2.T, dZ2)  # Gradient of hidden layer
        dZ1 = dHidden * (hidden_input > 0) # ReLU gradient for hidden layer. 
        dW1 = np.dot(dZ1, X_train) + lambda_value * W1 #calculating gradient w.r.t. W1 (including L2 regularization)
        dB1 = np.sum(dZ1, axis=1, keepdims=True) # Gradient w.r.t. B1 while Keepdims to ensure correct shape for broadcasting in numpy 
        
        
        #gradient descent updating
        W2 -= lr * dW2 
        B2 -= lr * dB2 
        W1 -= lr * dW1 
        B1 -= lr * dB1 
        # Every 100 epochs, print validation accuracy, training loss and epoch
        if DEBUG and epoch % 100 == 0:
            val_acc = evaluate_model(X_val, y_val, W1, B1, W2, B2)
            print(f"Epoch {epoch}: train_loss={total_loss:.4f}, val_acc={val_acc:.4f}")

    return (W1, B1, W2, B2) # return trained weights and biases

# Model Evaluation Function (returns only accuracy). this function is used to get the final accuracy of the model for each hyperparameter combination.
# this function returns the accuracy of the model.
def evaluate_model(X, y, W1, B1, W2, B2):
    hidden_input = np.dot(W1, X.T) + B1 # Linear input layer to hidden layer
    hidden_output = np.maximum(0, hidden_input) # ReLU activation for hidden layer
    output_input = np.dot(W2, hidden_output) + B2 # Linear output layer from hidden layer
    output_prob = 1 / (1 + np.exp(-output_input)) # Sigmoid activation for output layer
    output_prob = np.clip(output_prob, 1e-10, 1-1e-10) # Clip to prevent log(0)
    predictions = (output_prob > 0.5).astype(int) # Convert to binary predictions (0 or 1)
    return accuracy_score(y, predictions.flatten())

# Model Evaluation Function (returns only AUC) this function is used to test the model on the validation set and select the best model based on AUC.
# this function returns the AUC of the model.
def evaluate_auc(X, y, W1, B1, W2, B2): 
    hidden_input = np.dot(W1, X.T) + B1 # Linear input layer to hidden layer
    hidden_output = np.maximum(0, hidden_input) # ReLU activation for hidden layer
    output_input = np.dot(W2, hidden_output) + B2 # Linear output layer from hidden layer
    output_prob = 1 / (1 + np.exp(-output_input)) # Sigmoid activation for output layer
    output_prob = np.clip(output_prob, 1e-10, 1-1e-10) # Clip to prevent log(0)
    # Flatten output and compute AUC
    return roc_auc_score(y, output_prob.flatten()) # Use flatten() to convert to 1D array

# Grid Search Function (using KFold on the training set)
def grid_search_fold(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights): # Add class_weights
    if DEBUG:
        print(f"Grid search fold: hidden_size={hidden_size}, lr={lr}")
    start_time = time()
    best_weights = train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights) # Add class_weights
    train_time = time() - start_time
    val_acc = evaluate_model(X_val, y_val, *best_weights) # Unpack weights tuple with * operator 
    val_auc = evaluate_auc(X_val, y_val, *best_weights) # Unpack weights tuple with * operator
    if DEBUG:
        print(f"Training time: {train_time:.2f}s, Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}")
    return val_acc, train_time, val_auc

# Initialize grid result matrices for TEST SET metrics
grid_test_acc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) #matrix for test accuracy
grid_test_auc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) #matrix for test AUC
grid_train_time = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) #matrix for training time
grid_test_acc = np.zeros((len(hidden_sizes), len(learning_rates)), dtype=np.float32) #matrix for test accuracy

#NOW WE PERFORM 5 FOLD CROSS VALIDATION FOR EACH HYPERPARAMETER COMBINATION AND SELECT THE BEST MODEL BASED ON VALIDATION AUC AND TEST IT ON THE TEST SET

# K-Fold Cross-Validation on the training set
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for hs_idx, hidden_size in enumerate(hidden_sizes):
    for lr_idx, lr in enumerate(learning_rates):
        best_fold_auc = -np.inf  # Track best AUC across folds
        best_fold_model = None # Track best model weights
        best_fold_time = 0 # Track best training time
        
        for train_idx, val_idx in cv.split(X_train_full):
            X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
            
            # Train model and get validation AUC
            start_time = time()
            model_weights = train_mlp(X_train, y_train, X_val, y_val, 
                                     hidden_size, lr, epochs, lambda_value, class_weights)
            train_time = time() - start_time
            val_auc = evaluate_auc(X_val, y_val, *model_weights)  # Use AUC for selection
            
            # Track best fold model for this hyperparameter combo
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_model = model_weights
                best_fold_time = train_time
        
        # After all folds, testing the BEST MODEL on TEST SET (using accuracy)
        test_acc = evaluate_model(X_test, y_test, *best_fold_model)  # Final accuracy metric
        
        # Updating grid results
        grid_test_acc[hs_idx, lr_idx] = test_acc
        grid_train_time[hs_idx, lr_idx] = best_fold_time
        
        print(f"Combo: hidden_size={hidden_size}, lr={lr:.2e} | "
              f"Test Acc: {test_acc:.4f}, Time: {best_fold_time:.2f}s")

# Heatmap for Test Accuracy (using AUC-selected models)
plt.figure(figsize=(12, 10))
sns.heatmap(grid_test_acc, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.1e}" for lr in learning_rates], 
            yticklabels=hidden_sizes, cmap='viridis', 
            annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
plt.title('Test Accuracy (Models Selected via Validation AUC)', fontsize=16)
plt.xlabel('Learning Rate', fontsize=14)
plt.ylabel('Hidden Size', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show(block=False)

# Heatmap for Training Time
plt.figure(figsize=(12, 10))
sns.heatmap(grid_train_time, annot=True, fmt=".1f", 
            xticklabels=[f"{lr:.1e}" for lr in learning_rates], 
            yticklabels=hidden_sizes, cmap='rocket', 
            annot_kws={"size": 12}, cbar_kws={"shrink": 0.8})
plt.title('Training Time (Seconds)', fontsize=16)
plt.xlabel('Learning Rate', fontsize=14)
plt.ylabel('Hidden Size', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show(block=False)
