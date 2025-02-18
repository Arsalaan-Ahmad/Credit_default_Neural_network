%% MLP for Default Payment Prediction with Grid Search, Early Stopping,
%% Class Weighting (via oversampling) and Ensemble, while Tracking Recall & F1

clear; clc; close all;

%% 1. Load and Prepare Data
data = readtable('C:\Users\arsal\OneDrive\Neural computing GW\default of credit card clients.xls');
dataArray = table2array(data);

% Extract features and target labels.
% (Assume last column is the binary target: 0 or 1)
X = dataArray(:, 1:end-1);  
Y = dataArray(:, end);      

% Transpose so that each column is a sample.
X = X';

% Convert targets to one-hot encoding.
% Class 0 becomes index 1 and class 1 becomes index 2.
T = full(ind2vec(Y' + 1));

%% 2. Outer Data Split: 80% Training, 20% Test
numSamples = size(X,2);
cvOuter = cvpartition(numSamples, 'HoldOut', 0.2);

xTrain = X(:, cvOuter.training);
tTrain = T(:, cvOuter.training);

xTest  = X(:, cvOuter.test);
tTest  = T(:, cvOuter.test);

%% 3. Grid Search Setup with Early Stopping and Class Weighting
% Candidate hyperparameters:
hiddenLayerSizes = [ 10, 15, 20 ];      % Candidate numbers of neurons
learningRates    = [ 0.15, 0.225, 0.25];  % Candidate learning rates

numFolds = 5;  % 5-fold CV for grid search.
bestValAcc = 0;
bestParams = struct('hiddenLayerSize', [], 'learningRate', []);

% To store grid search results.
% Columns: [hiddenLayerSize, learningRate, avgValAcc, avgValLoss, avgTrainLoss, ...
%           avgTrainTime, avgValTime, percentLossDiff, avgRecall, avgF1]
gridResults = [];

% Weight vector: for example, class0 -> 4, class1 -> 6.
weightVec = [1, 3];

fprintf('Starting Grid Search with Early Stopping & Class Weighting...\n');
for h = 1:length(hiddenLayerSizes)
    for l = 1:length(learningRates)
        currHiddenSize = hiddenLayerSizes(h);
        currLearnRate  = learningRates(l);
        
        % Create 5-fold CV partition on the training set.
        cvInner = cvpartition(size(xTrain,2), 'KFold', numFolds);
        
        % Preallocate arrays to store per-fold metrics.
        foldAcc       = zeros(numFolds, 1);
        foldValLosses = zeros(numFolds, 1);
        foldTrainLoss = zeros(numFolds, 1);
        foldTrainTimes = zeros(numFolds, 1);
        foldValTimes   = zeros(numFolds, 1);
        foldRecall    = zeros(numFolds, 1);
        foldF1        = zeros(numFolds, 1);
        
        for fold = 1:cvInner.NumTestSets
            % Get indices for training and external validation.
            trainIdx = cvInner.training(fold);
            valIdx   = cvInner.test(fold);
            
            % Use only the training fold for oversampling.
            xTrainFold = xTrain(:, trainIdx);
            tTrainFold = tTrain(:, trainIdx);
            [xTrainFoldW, tTrainFoldW] = applyClassWeight(xTrainFold, tTrainFold, weightVec);
            
            % Create and configure the network.
            net = patternnet(currHiddenSize);
            net.trainFcn = 'traingd';  % Gradient descent backpropagation.
            net.trainParam.lr = currLearnRate;
            % Use internal division (80%/20%) for early stopping on oversampled data.
            net.divideFcn = 'dividerand';
            net.trainParam.epochs = 500;
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio   = 0.2;
            net.divideParam.testRatio  = 0;
            net.trainParam.max_fail = 6;
            
            % Train the network on oversampled training fold.
            tStartTrain = tic;
            [net, tr] = train(net, xTrainFoldW, tTrainFoldW);
            foldTrainTimes(fold) = toc(tStartTrain);
            
            % Compute training loss on the internal training subset.
            trainInd = tr.trainInd;
            trainOutput = net(xTrainFoldW(:, trainInd));
            lossTrain = perform(net, tTrainFoldW(:, trainInd), trainOutput);
            foldTrainLoss(fold) = lossTrain;
            
            % Evaluate on external validation fold (non-oversampled).
            tStartVal = tic;
            externalOutput = net(xTrain(:, valIdx));
            foldValTimes(fold) = toc(tStartVal);
            
            lossVal = perform(net, tTrain(:, valIdx), externalOutput);
            foldValLosses(fold) = lossVal;
            predVal = vec2ind(externalOutput);
            trueVal = vec2ind(tTrain(:, valIdx));
            accVal  = sum(predVal == trueVal) / numel(trueVal);
            foldAcc(fold) = accVal;
            
            % Compute confusion matrix, recall, and F1 for minority class (class 1).
            C = confusionmat(trueVal, predVal);
            if size(C,1) < 2
                C(2,2) = 0;
            end
            if sum(C(2,:)) > 0
                recall = C(2,2) / sum(C(2,:));
            else
                recall = 0;
            end
            if sum(C(:,2)) > 0
                precision = C(2,2) / sum(C(:,2));
            else
                precision = 0;
            end
            if (precision + recall) > 0
                f1 = 2 * (precision * recall) / (precision + recall);
            else
                f1 = 0;
            end
            foldRecall(fold) = recall;
            foldF1(fold) = f1;
        end  % end of inner fold loop
        
        % Average the metrics over the folds.
        avgAcc       = mean(foldAcc);
        avgValLoss   = mean(foldValLosses);
        avgTrainLoss = mean(foldTrainLoss);
        avgTrainTime = mean(foldTrainTimes);
        avgValTime   = mean(foldValTimes);
        percentDiff  = ((avgValLoss - avgTrainLoss) / avgTrainLoss) * 100;
        avgRecall    = mean(foldRecall);
        avgF1        = mean(foldF1);
        
        % Append results for this hyperparameter combination.
        gridResults = [gridResults; currHiddenSize, currLearnRate, avgAcc, avgValLoss, avgTrainLoss, avgTrainTime, avgValTime, percentDiff, avgRecall, avgF1];
        
        % Print the results for this combination.
        fprintf('HiddenLayerSize = %d, LearningRate = %.4f\n', currHiddenSize, currLearnRate);
        fprintf('  Avg Val Accuracy         = %.2f%%\n', avgAcc*100);
        fprintf('  Avg Val Loss             = %.4f\n', avgValLoss);
        fprintf('  Avg Train Loss           = %.4f\n', avgTrainLoss);
        fprintf('  Percent Loss Diff        = %.2f%%\n', percentDiff);
        fprintf('  Avg Training Time        = %.4fs\n', avgTrainTime);
        fprintf('  Avg Val Prediction Time  = %.4fs\n', avgValTime);
        fprintf('  Avg Recall (class 1)       = %.2f%%\n', avgRecall*100);
        fprintf('  Avg F1 Score (class 1)     = %.2f%%\n\n', avgF1*100);
        
        % Update best parameters if current average validation accuracy is higher.
        if avgAcc > bestValAcc
            bestValAcc = avgAcc;
            bestParams.hiddenLayerSize = currHiddenSize;
            bestParams.learningRate = currLearnRate;
        end
    end
end

fprintf('\nBest parameters from grid search: HiddenLayerSize = %d, LearningRate = %.4f (Val Accuracy = %.2f%%)\n', ...
    bestParams.hiddenLayerSize, bestParams.learningRate, bestValAcc*100);

%% 4. Final Ensemble Model with Best Hyperparameters
% Build ensemble models using a new 5-fold CV on the training set.
finalCV = cvpartition(size(xTrain,2), 'KFold', numFolds);
ensembleModels = cell(numFolds, 1);

fprintf('\nTraining final ensemble models with best hyperparameters...\n');
for fold = 1:finalCV.NumTestSets
    % Use the training partition for each fold.
    trainIdx = finalCV.training(fold);
    xFold = xTrain(:, trainIdx);
    tFold = tTrain(:, trainIdx);
    
    % Apply oversampling on the training fold.
    [xFoldW, tFoldW] = applyClassWeight(xFold, tFold, weightVec);
    
    netFinal = patternnet(bestParams.hiddenLayerSize);
    netFinal.trainFcn = 'traingd';
    netFinal.trainParam.lr = bestParams.learningRate;
    netFinal.divideFcn = 'dividerand';
    netFinal.divideParam.trainRatio = 0.8;
    netFinal.divideParam.valRatio   = 0.2;
    netFinal.divideParam.testRatio  = 0;
    netFinal.trainParam.max_fail = 6;
    
    [netFinal, ~] = train(netFinal, xFoldW, tFoldW);
    ensembleModels{fold} = netFinal;
    fprintf('  Fold %d model trained.\n', fold);
end

%% 5. Ensemble Prediction on the Held-Out Test Set (original distribution)
ensembleOutput = zeros(size(tTest));  % Preallocate (numClasses x numTestSamples)
for fold = 1:numFolds
    modelOutput = ensembleModels{fold}(xTest);
    ensembleOutput = ensembleOutput + modelOutput;
end
ensembleOutput = ensembleOutput / numFolds;  % Average the outputs

predEnsemble = vec2ind(ensembleOutput);
trueTest     = vec2ind(tTest);
ensembleAcc  = sum(predEnsemble == trueTest) / numel(trueTest);
ensembleLoss = perform(ensembleModels{1}, tTest, ensembleOutput);  % Use one model's settings

% Compute confusion matrix for test set, then recall and F1 for minority class (class 1).
C_test = confusionmat(trueTest, predEnsemble);
if size(C_test,1) < 2
    C_test(2,2) = 0;
end
if sum(C_test(2,:)) > 0
    testRecall = C_test(2,2) / sum(C_test(2,:));
else
    testRecall = 0;
end
if sum(C_test(:,2)) > 0
    testPrecision = C_test(2,2) / sum(C_test(:,2));
else
    testPrecision = 0;
end
if (testPrecision + testRecall) > 0
    testF1 = 2 * (testPrecision * testRecall) / (testPrecision + testRecall);
else
    testF1 = 0;
end

fprintf('\n=== Final Ensemble Performance on Test Set ===\n');
fprintf('Test Loss: %.4f, Test Accuracy: %.2f%%\n', ensembleLoss, ensembleAcc*100);
fprintf('Test Recall (class 1): %.2f%%, Test F1 Score (class 1): %.2f%%\n', testRecall*100, testF1*100);

%% ---- Helper Function: applyClassWeight ----
function [x_weighted, t_weighted] = applyClassWeight(x_in, t_in, weightVec)
    % This function oversamples each sample in x_in according to its class weight.
    % x_in: feature matrix (each column is a sample)
    % t_in: one-hot encoded targets
    % weightVec: a vector of weights for each class (e.g., [4, 6])
    % The output x_weighted and t_weighted contain the oversampled data.
    
    x_weighted = [];
    t_weighted = [];
    
    numSamples = size(x_in, 2);
    for i = 1:numSamples
        classIdx = vec2ind(t_in(:, i));  % returns 1 (for class 0) or 2 (for class 1)
        w = weightVec(classIdx);
        % Replicate the sample w times.
        x_weighted = [x_weighted, repmat(x_in(:, i), 1, w)];
        t_weighted = [t_weighted, repmat(t_in(:, i), 1, w)];
    end
end
