%% Initialization
clear; clc; close all;
rng(42, 'twister'); %to make it similar to python

%% Load and preprocess data
data = readtable('default of credit card clients.xls');
data = rmmissing(data);  % Remove missing values

% Convert to arrays
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));
y = logical(y);

%% Train-test split
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train_full = X(cv.training,:);
y_train_full = y(cv.training);
X_test = X(cv.test,:);
y_test = y(cv.test);

%% Standardization
[Z, mu, sigma] = zscore(X_train_full);
X_train_full = Z;
X_test = (X_test - mu) ./ sigma;

%% Class weights
class_weights = calc_class_weights(y_train_full);
fprintf('Class distribution (train): %.2f%% positive\n', 100*mean(y_train_full));
fprintf('Class weights: [%.4f, %.4f]\n', class_weights(1), class_weights(2));

%% Hyperparameters
hidden_sizes = [12,17,24,32,48];
lambda_value = 0.001;
epochs = 500;
DEBUG = true;
learning_rates= [0.001, 0.00560932, 0.03146442, 0.17649389, 0.99001];%calculated on python

%% Grid Search Setup
grid_test_acc = zeros(length(hidden_sizes), length(learning_rates)); % matrix for storing test accuracies for every hyperparameter combinations
grid_train_time = zeros(length(hidden_sizes), length(learning_rates)); %matrix for storing  trainig time for every hyperparameter combinations

%% K-Fold Cross Validation
cv = cvpartition(y_train_full, 'KFold', 5);
for hs_idx = 1:length(hidden_sizes)
    hidden_size = hidden_sizes(hs_idx);
    
    for lr_idx = 1:length(learning_rates)
        lr = learning_rates(lr_idx);
        best_fold_auc = -inf; %store best fold AUC value
        best_fold_model = struct(); %store best fold model's model weight 
        best_fold_time = 0; %store best fold model's training time
        
        for fold = 1:cv.NumTestSets
            % Split data into training and validation sets
            trainIdx = cv.training(fold);
            testIdx = cv.test(fold);
            X_train = X_train_full(trainIdx,:);
            y_train = y_train_full(trainIdx);
            X_val = X_train_full(testIdx,:);
            y_val = y_train_full(testIdx);
            
            % Train model
            tic;
            model = train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda_value, class_weights);
            train_time = toc; %measuring training time
            
            % Evaluate AUC
            val_auc = evaluate_auc(X_val, y_val, model);
            
            % Track best model
            if val_auc > best_fold_auc
                best_fold_auc = val_auc;
                best_fold_model = model;
                best_fold_time = train_time;
            end
        end
        
        % Test best model
        test_acc = evaluate_acc(X_test, y_test, best_fold_model);
        grid_test_acc(hs_idx, lr_idx) = test_acc;
        grid_train_time(hs_idx, lr_idx) = best_fold_time;
        
        fprintf('Combo: hidden_size=%d, lr=%.2e | Test Acc: %.4f, Time: %.2fs\n', ...
                hidden_size, lr, test_acc, best_fold_time);
    end
end

%% Plot Heatmaps
% Test Accuracy
figure;
heatmap(learning_rates, hidden_sizes, grid_test_acc, 'Colormap', parula, ...
        'ColorbarVisible', 'on', 'CellLabelFormat', '%.3f');
title('Test Accuracy (Models Selected via Validation AUC)');
xlabel('Learning Rate');
ylabel('Hidden Size');

% Training Time
figure;
heatmap(learning_rates, hidden_sizes, grid_train_time, 'Colormap', hot, ...
        'ColorbarVisible', 'on', 'CellLabelFormat', '%.1f');
title('Training Time (Seconds)');
xlabel('Learning Rate');
ylabel('Hidden Size');

%% Helper Functions 
% this function is used to change the class weight inorder to deal with the
% imbalence data.
% we are doing balenced weight distirbution i.e-t each class gets a weight
% that is inversely proportional to its frequency in the dataset.
function weights = calc_class_weights(y)
    n_samples = length(y);
    n_classes = 2;
    bincounts = [sum(~y), sum(y)];
    weights = n_samples ./ (n_classes * bincounts);
end

%%  train_mlp Function
% here weight for input to hidden layer. Initialized using He initialization
% b1: bias for hidden layer
% w2: weight from hidden to output layer. Same initialization as w1.
% b2: bias for output layer
%  weight_vector is calculated to implement weight loss function which
%  gives different importance to class based on class weight which we take
%  in as input which we just calculated above.

function model = train_mlp(X_train, y_train, X_val, y_val, hidden_size, lr, epochs, lambda, class_weights)
    % Transpose FIRST to get correct dimensions
    X_train = X_train';  % Now [input_size x n_samples]
    [input_size, n_samples] = size(X_train);
    
    % He initialization of w1 and w2 and initialization of b1 and b2 to zero
    he_scale = @(fan_in) sqrt(2/fan_in);
    W1 = randn(hidden_size, input_size) * he_scale(input_size);
    B1 = zeros(hidden_size, 1);
    W2 = randn(1, hidden_size) * he_scale(hidden_size);
    B2 = 0;
    
    % Create weight vector 
    weight_vector = (class_weights(1)*(~y_train) + class_weights(2)*y_train)';
    
    for epoch = 1:epochs

        % Forward pass 
        % compute input to the hiden layer 'hidden_input'
        % hiden layer: applying ReLU activation
        % compute input to the output layer 'output_input'
        % output layer: applying sigmoid function and clipping to prevent log(0)
        hidden = max(0, W1*X_train + B1);
        output = 1./(1 + exp(-(W2*hidden + B2)));
        output = max(min(output, 1-1e-10), 1e-10);
        
        % Loss calculation
        % we are using weighted binary cross-entropy loss function as we are performing binary classification
        % L2 regularization term is added to prevent overfitting
        data_loss = -mean((y_train'.*log(output) + (1-y_train').*log(1 - output)) .* weight_vector);
        reg_loss = 0.5*lambda*(sum(W1(:).^2) + sum(W2(:).^2));
        total_loss = data_loss + reg_loss;
        
        % performing Backprop
        dZ2 = (output - y_train') .* weight_vector / n_samples; %calculating  gradient of loss w.r.t. output layer
        W2_grad = dZ2*hidden' + lambda*W2;  %calculating gradient w.r.t. W2 (including L2 regularization)
        B2_grad = sum(dZ2, 2); %Gradient w.r.t. B2 
        dHidden = (W2'*dZ2) .* (hidden > 0);  % Gradient of hidden layer
        W1_grad = dHidden*X_train' + lambda*W1; % calculating gradient w.r.t. W1 (including L2 regularization)
        B1_grad = sum(dHidden, 2); %Gradient w.r.t. B1 
        
        % updating the parameters
        W2 = W2 - lr*W2_grad;
        B2 = B2 - lr*B2_grad;
        W1 = W1 - lr*W1_grad;
        B1 = B1 - lr*B1_grad;
        
        % Using X_val/y_val for validation 
        if exist('DEBUG', 'var') && DEBUG && mod(epoch,100) == 0
            val_acc = evaluate_acc(X_val', y_val, struct('W1',W1,'B1',B1,'W2',W2,'B2',B2));
            fprintf('Epoch %4d: loss=%.4f, val_acc=%.4f\n', epoch, total_loss, val_acc);
        end
    end
    
    model.W1 = W1;
    model.B1 = B1;
    model.W2 = W2;
    model.B2 = B2;
end

%% evaluate_acc Function
% we use this function to calculate the accuracy of the best fold model
% for each hyper parameter combo on the test set.
function acc = evaluate_acc(X, y, model)
    X = X';  % Transpose to [features x samples]
    hidden = max(0, model.W1*X + model.B1);
    output = 1./(1 + exp(-(model.W2*hidden + model.B2)));
    predictions = output > 0.5;
    acc = mean(predictions == y');
end
%% AUC evaluation function
% we use this function to calculate model auc which we use to find the best
% model for every hyperparameter combination.
function auc = evaluate_auc(X, y, model)
    X = X';
    hidden = max(0, model.W1*X + model.B1);
    output = 1 ./ (1 + exp(-(model.W2*hidden + model.B2)));
    [~,~,~,auc] = perfcurve(y, output', true);
end