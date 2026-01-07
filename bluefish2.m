% 🌊 BlueFish v2.0 (Combined)
% ================================================================
% This script combines two modes:
% 1. EXPERIMENT MODE: Runs the 10-fold CV for your paper's results
%    (Table 1).
% 2. DEMO MODE: Trains the BEST model (Random Forest) on 100%
%    of the data and lets you identify a single file.
%
% ⚠️ v2.0 Changelog:
%   - Uses rich statistical features (mean, std, skew, kurt).
%   - Data Cleaning step is standard.
%   - EXPERIMENT mode uses 10-fold CV for robust results.
%   - DEMO mode now trains the *Random Forest* (your proven
%     winner), not ECOC, for best performance.
%
clc; clear; close all;
fprintf('🌊 BlueFish v2.0 (Combined) - Marine Species Identification\n');
fprintf('==============================================================\n');

% --- Step 1: Mode Selection ---
modeChoice = questdlg('Select Operation Mode', ...
                      'BlueFish v2.0', ...
                      'Run Experiment (for Paper)', 'Run Demo (Test Single File)', ...
                      'Run Experiment (for Paper)');
if isempty(modeChoice)
    fprintf('❌ No mode selected. Exiting.\n');
    return;
end

% --- Step 2: Load Data (Common to both modes) ---
datasetPath = uigetdir(pwd, '📁 Select the dataset folder (each species = subfolder)');
if datasetPath == 0
    error('❌ No folder selected. Exiting.');
end
subfolders = dir(datasetPath);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.'));
if isempty(subfolders)
    error('❌ No subfolders found in dataset folder.');
end
fprintf('🔍 Found %d species folders.\n', numel(subfolders));

% --- Step 3: Feature Extraction (Common to both modes) ---
features = [];
labels = [];
fprintf('🎧 Extracting statistical features (mean, std, skew, kurt)...\n');
for k = 1:numel(subfolders)
    folder = fullfile(datasetPath, subfolders(k).name);
    audioFiles = dir(fullfile(folder, '*.wav'));
    fprintf('  🐠 Species: %s (%d files)\n', subfolders(k).name, numel(audioFiles));
    
    for i = 1:numel(audioFiles)
        filePath = fullfile(folder, audioFiles(i).name);
        try
            [y, fs] = audioread(filePath);
            y = y(:,1); % mono
            y = y / max(abs(y) + eps);
            
            coeffs = mfcc(y, fs, 'NumCoeffs', 13);
            
            % Rich statistical features (1x56 or 1x52)
            stat_features = [mean(coeffs, 1), ...
                            std(coeffs, 0, 1), ...
                            skewness(coeffs, 1, 1), ...
                            kurtosis(coeffs, 1, 1)];
                        
            features = [features; stat_features];
            labels = [labels; string(subfolders(k).name)];
        catch ME
            warning("⚠ Feature extraction failed for '%s': %s", audioFiles(i).name, ME.message);
        end
    end
end
fprintf('✅ Feature extraction complete. Total samples: %d\n', numel(labels));
fprintf('  Feature vector size: 1x%d\n', size(features, 2));

% --- Step 4: Data Cleaning (Common to both modes) ---
fprintf('🧹 Cleaning data...\n');
minSamples = 5; 
[uniqueLabels, ~, labelIdx] = unique(labels);
labelCounts = accumarray(labelIdx, 1);
labelsToKeep = uniqueLabels(labelCounts >= minSamples);
if numel(labelsToKeep) < numel(uniqueLabels)
    fprintf('  ⚠ WARNING: Removing %d classes with fewer than %d samples.\n', ...
            numel(uniqueLabels) - numel(labelsToKeep), minSamples);
else
    fprintf('  All classes have %d or more samples. No classes removed.\n', minSamples);
end
keepIdx = ismember(labels, labelsToKeep);
features = features(keepIdx, :);
labels = labels(keepIdx);
cleanedLabels = categorical(labels); % Use categorical for classification
fprintf('✅ Data cleaning complete. Remaining samples for use: %d\n', numel(cleanedLabels));
if isempty(cleanedLabels)
    error('❌ No data remaining after cleaning. Check dataset or lower minSamples threshold.');
end

% --- Step 5: Execute Selected Mode ---
switch modeChoice
    case 'Run Experiment (for Paper)'
        % === EXPERIMENT MODE ===
        % This code generates the data for your paper's results section.
        fprintf('\n--- 🔬 EXECUTING 10-FOLD CV EXPERIMENT ---\n');
        
        k = 10;
        cv = cvpartition(cleanedLabels, 'KFold', k);
        accuracies_ecoc = zeros(k, 1);
        accuracies_knn = zeros(k, 1);
        accuracies_rf = zeros(k, 1);

        allTestLabels = [];
        allPredLabels_rf = []; 

        % --- Define the (non-optimized) classifiers ---
        fprintf('Defining models with standard parameters...\n');
        ecocTemplate = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
        ecocClassifier = @(trainFeatures, trainLabels) ...
            fitcecoc(trainFeatures, trainLabels, 'Learners', ecocTemplate);
        knnClassifier = @(trainFeatures, trainLabels) ...
            fitcknn(trainFeatures, trainLabels, 'NumNeighbors', 5);
        rfTemplate = templateTree('Reproducible', true);
        rfClassifier = @(trainFeatures, trainLabels) ...
            fitcensemble(trainFeatures, trainLabels, 'Method', 'Bag', ...
            'NumLearningCycles', 100, 'Learners', rfTemplate);
        
        fprintf('🔬 Running 10-fold benchmark...\n');
        for i = 1:k
            fprintf('  --- Fold %d/%d ---\n', i, k);
            trainIdx = training(cv, i);
            testIdx = test(cv, i);
            
            trainingFeatures = features(trainIdx, :);
            trainingLabels = cleanedLabels(trainIdx);
            testFeatures = features(testIdx, :);
            testLabels = cleanedLabels(testIdx);

            % --- Classifier 1: ECOC ---
            fprintf('    Training ECOC...\n');
            model_ecoc = ecocClassifier(trainingFeatures, trainingLabels);
            predictedLabels_ecoc = predict(model_ecoc, testFeatures);
            accuracies_ecoc(i) = sum(predictedLabels_ecoc == testLabels) / numel(testLabels);
            fprintf('    ✅ ECOC Fold Accuracy: %.2f%%\n', accuracies_ecoc(i) * 100);

            % --- Classifier 2: KNN ---
            fprintf('    Training KNN (k=5)...\n');
            model_knn = knnClassifier(trainingFeatures, trainingLabels);
            predictedLabels_knn = predict(model_knn, testFeatures);
            accuracies_knn(i) = sum(predictedLabels_knn == testLabels) / numel(testLabels);
            fprintf('    ✅ KNN Fold Accuracy: %.2f%%\n', accuracies_knn(i) * 100);

            % --- Classifier 3: Random Forest ---
            fprintf('    Training Random Forest (100 trees)...\n');
            model_rf = rfClassifier(trainingFeatures, trainingLabels);
            predictedLabels_rf = predict(model_rf, testFeatures);
            accuracies_rf(i) = sum(predictedLabels_rf == testLabels) / numel(testLabels);
            fprintf('    ✅ Random Forest Fold Accuracy: %.2f%%\n', accuracies_rf(i) * 100);
            
            allTestLabels = [allTestLabels; testLabels];
            allPredLabels_rf = [allPredLabels_rf; predictedLabels_rf];
        end

        % --- Report Final Results ---
        fprintf('\n--- ✅ FINAL RESULTS (10-FOLD CV) ---\n');
        fprintf('  Classifier \t\t | Mean Accuracy \t | Std. Deviation\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('  Random Forest (100T) \t | %.2f%% \t\t | %.2f\n', mean(accuracies_rf) * 100, std(accuracies_rf) * 100);
        fprintf('  ECOC (Gaussian SVM) \t | %.2f%% \t\t | %.2f\n', mean(accuracies_ecoc) * 100, std(accuracies_ecoc) * 100);
        fprintf('  KNN (k=5) \t\t | %.2f%% \t\t | %.2f\n', mean(accuracies_knn) * 100, std(accuracies_knn) * 100);
        fprintf('\n👉 Use these mean accuracies and std. deviations for your results table!\n');

        fprintf('\n📈 Generating confusion matrix for best model (Random Forest)...\n');
        figure('Name', 'Random Forest Confusion Matrix (10-Fold CV)', 'NumberTitle', 'off');
        confusionchart(allTestLabels, allPredLabels_rf, ...
            'Title', sprintf('Random Forest (100 Trees) (Mean Acc: %.2f%%)', mean(accuracies_rf) * 100), ...
            'RowSummary', 'row-normalized', ...
            'ColumnSummary', 'column-normalized');
        
        fprintf('\n✅ Experiment complete.\n');

        
    case 'Run Demo (Test Single File)'
        % === DEMO MODE ===
        % This trains the BEST model (Random Forest) on 100%
        % of the cleaned data for a final, usable application.
        fprintf('\n--- 🎶 EXECUTING DEMO MODE ---\n');
        
        % 1. Train Classifier on 100% of *cleaned* data
        fprintf('🤖 Training BEST model (Random Forest) on ALL %d cleaned samples...\n', numel(cleanedLabels));
        
        rfTemplate_demo = templateTree('Reproducible', true);
        model_rf_demo = fitcensemble(features, cleanedLabels, 'Method', 'Bag', ...
            'NumLearningCycles', 100, 'Learners', rfTemplate_demo);
        
        fprintf('🤖 Model trained with %d samples across %d species.\n', numel(cleanedLabels), numel(unique(cleanedLabels)));
        
        % 2. Save Model
        save('bluefish_demo_model_rf.mat', 'model_rf_demo', 'cleanedLabels');
        fprintf('💾 Model saved as bluefish_demo_model_rf.mat\n');
        
        % 3. Test Phase
        [file, path] = uigetfile({'*.wav'}, '🎙 Select an audio file to identify');
        if isequal(file,0)
            disp('❌ No file selected. Exiting.');
            return;
        end
        
        testPath = fullfile(path, file);
        fprintf('\n🔊 Testing file: %s\n', file);
        
        try
            [yTest, fsTest] = audioread(testPath);
            yTest = yTest(:,1);
            yTest = yTest / max(abs(yTest) + eps);
            
            % 4. Extract Features and Predict
            % MUST use the *exact same* feature extraction
            coeffsTest = mfcc(yTest, fsTest, 'NumCoeffs', 13);
            stat_features_test = [mean(coeffsTest, 1), ...
                                  std(coeffsTest, 0, 1), ...
                                  skewness(coeffsTest, 1, 1), ...
                                  kurtosis(coeffsTest, 1, 1)];
                              
            predictedSpecies = predict(model_rf_demo, stat_features_test);
            
            if iscell(predictedSpecies)
                predictedSpecies = predictedSpecies{1};
            end
            fprintf('✅ Detected Species: %s\n', string(predictedSpecies));
            
            % 5. Visualize Spectrogram
            figure('Name', 'Marine Audio Spectrogram (Demo)', 'NumberTitle', 'off');
            spectrogram(yTest, 256, [], [], fsTest, 'yaxis');
            title(sprintf('Detected: %s', string(predictedSpecies)));
            colormap jet;
            colorbar;
            
            % 6. Play Audio
            fprintf('🎵 Playing detected audio...\n');
            sound(yTest, fsTest);
            pause(length(yTest) / fsTest); % Wait for playback
            fprintf('🪸 Audio playback complete.\n');
            
        catch ME
            error("❌ Failed to process test file '%s': %s", file, ME.message);
        end
end
fprintf('\n🌊 BlueFish v2.0 Execution Complete!\n');