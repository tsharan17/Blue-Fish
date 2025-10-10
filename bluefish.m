% =========================================================================
%       MARINE SPECIES IDENTIFICATION USING MACHINE LEARNING (SVM + MFCC)
% =========================================================================
clear; clc; close all;

fprintf('\n🌊 Marine Species Identification using ML (MFCC + SVM)\n');
fprintf('=========================================================\n');

%% --- Step 1: Select Dataset Folder ---
datasetPath = uigetdir('C:\', 'Select Marine Species Dataset Folder');
if datasetPath == 0
    disp('❌ No folder selected. Exiting...');
    return;
end

fprintf('📂 Loading dataset from: %s\n', datasetPath);

%% --- Step 2: Load Audio Files and Extract MFCC Features ---
audioDS = audioDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numFiles = numel(audioDS.Files);
fprintf('✅ Found %d audio files.\n', numFiles);

if numFiles < 2
    error('❌ Need at least 2 audio files to train the model.');
end

allFeatures = [];
allLabels = [];

for i = 1:numFiles
    try
        [audioIn, fs] = audioread(audioDS.Files{i});
        if size(audioIn,2) > 1
            audioIn = mean(audioIn,2); % Convert stereo → mono
        end
        audioIn = audioIn - mean(audioIn); % Remove DC offset

        % --- Extract MFCC Features ---
        coeffs = mfcc(audioIn, fs, 'NumCoeffs', 20);
        meanMFCC = mean(coeffs);  % Average across time frames

        allFeatures = [allFeatures; meanMFCC];
        allLabels = [allLabels; audioDS.Labels(i)];

        fprintf('🎵 Processed: %s\n', audioDS.Files{i});
    catch ME
        warning('⚠ Could not read %s: %s', audioDS.Files{i}, ME.message);
    end
end

fprintf('\n✅ Feature extraction complete for %d files.\n', size(allFeatures,1));

%% --- Step 3: Split Data into Train/Test (Safe Version) ---
numClasses = numel(unique(allLabels));

if numFiles < 10 || numClasses < 2
    warning('⚠ Too few samples for test split. Using all data for training.');
    Xtrain = allFeatures;
    Ytrain = allLabels;
    Xtest = allFeatures;
    Ytest = allLabels;
else
    try
        % Try stratified split first
        cv = cvpartition(allLabels, 'HoldOut', 0.2);
    catch
        % Fall back to non-stratified split
        warning('⚠ Using non-stratified split due to small or unbalanced dataset.');
        cv = cvpartition(numel(allLabels), 'HoldOut', 0.2);
    end

    Xtrain = allFeatures(training(cv), :);
    Ytrain = allLabels(training(cv));
    Xtest  = allFeatures(test(cv), :);
    Ytest  = allLabels(test(cv));
end

fprintf('📊 Training samples: %d | Testing samples: %d\n', ...
    size(Xtrain,1), size(Xtest,1));

%% --- Step 4: Train SVM Classifier ---
fprintf('\n🚀 Training SVM model (RBF kernel)...\n');
svmTemplate = templateSVM('KernelFunction','rbf','KernelScale','auto');
svmModel = fitcecoc(Xtrain, Ytrain, 'Coding', 'onevsall', 'Learners', svmTemplate);

fprintf('✅ Training complete.\n');

%% --- Step 5: Evaluate Model (if possible) ---
if size(Xtest,1) > 1 && numClasses > 1
    predictedLabels = predict(svmModel, Xtest);
    accuracy = mean(predictedLabels == Ytest) * 100;
    fprintf('🎯 Test Accuracy: %.2f%%\n', accuracy);

    figure;
    confusionchart(Ytest, predictedLabels);
    title('Confusion Matrix - Marine Species Identification');
else
    fprintf('⚠ Skipping accuracy test (not enough data for split).\n');
end

%% --- Step 6: Save the Trained Model ---
save('MarineSpecies_SVM_Model.mat', 'svmModel', 'audioDS');
fprintf('💾 Model saved as MarineSpecies_SVM_Model.mat\n');

%% --- Step 7: Test a New Audio File ---
choice = questdlg('Do you want to test a new audio file?', ...
    'Test New File', 'Yes', 'No', 'Yes');

if strcmp(choice, 'Yes')
    [file, path] = uigetfile({'.mp3;.wav'}, 'Select Test Audio File');
    if isequal(file,0)
        disp('❌ No test file selected.');
    else
        testPath = fullfile(path, file);
        fprintf('\n🔍 Analyzing test audio: %s\n', testPath);

        [y, fs] = audioread(testPath);
        if size(y,2) > 1, y = mean(y,2); end
        y = y - mean(y);

        coeffsTest = mfcc(y, fs, 'NumCoeffs', 20);
        meanMFCC_test = mean(coeffsTest);

        predicted = predict(svmModel, meanMFCC_test);
        fprintf('\n🐋 Predicted Marine Species: %s\n', string(predicted));
    end
end

fprintf('\n=========================================================\n');
fprintf('Analysis Complete ✅\n');