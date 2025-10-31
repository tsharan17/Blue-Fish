% 🌊 BlueFish - Marine Acoustic Species Identification System
% ---------------------------------------------------------------
% ✅ Trains on labeled subfolders of marine species audio samples
% ✅ Lets you test a real audio file for species identification
% ✅ Fixed prediction output issue (cell/string compatibility)
% ✅ Requires Audio Toolbox

clc; clear; close all;
fprintf('🌊 BlueFish v1.0 - Marine Acoustic Species Identification System\n');
fprintf('==============================================================\n');

% === Step 1: Dataset Selection ===
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

% === Step 2: Feature Extraction ===
features = [];
labels = [];

fprintf('🎧 Extracting features...\n');
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

            % Extract MFCCs
            coeffs = mfcc(y, fs, 'NumCoeffs', 13);
            meanCoeffs = mean(coeffs, 1);

            features = [features; meanCoeffs];
            labels = [labels; string(subfolders(k).name)];
        catch ME
            warning("⚠️ Feature extraction failed for '%s': %s", audioFiles(i).name, ME.message);
        end
    end
end

fprintf('✅ Feature extraction complete. Training model...\n');

% === Step 3: Train Classifier ===
speciesClassifier = fitcecoc(features, labels);
fprintf('🤖 Model trained with %d samples across %d species.\n', numel(labels), numel(unique(labels)));

% === Step 4: Save Model ===
save('bluefish_model.mat', 'speciesClassifier', 'labels');
fprintf('💾 Model saved as bluefish_model.mat\n');

% === Step 5: Test Phase ===
choice = questdlg('Do you want to test a real audio file?', ...
                  'Test BlueFish', 'Yes', 'No', 'Yes');

if strcmp(choice, 'Yes')
    [file, path] = uigetfile({'*.wav'}, '🎙 Select an audio file to identify');
    if isequal(file,0)
        disp('❌ No file selected. Exiting.');
        return;
    end

    testPath = fullfile(path, file);
    fprintf('\n🔊 Testing file: %s\n', file);

    [yTest, fsTest] = audioread(testPath);
    yTest = yTest(:,1);
    yTest = yTest / max(abs(yTest) + eps);
    
    testMFCC = mfcc(yTest, fsTest, 'NumCoeffs', 13);
    meanTest = mean(testMFCC, 1);

    % === Step 6: Predict ===
    predictedSpecies = predict(speciesClassifier, meanTest);

    % Fix for cell output
    if iscell(predictedSpecies)
        predictedSpecies = predictedSpecies{1};
    end

    fprintf('✅ Detected Species: %s\n', string(predictedSpecies));

    % === Step 7: Visualize Spectrogram ===
    figure('Name', 'Marine Audio Spectrogram', 'NumberTitle', 'off');
    spectrogram(yTest, 256, [], [], fsTest, 'yaxis');
    title(sprintf('Detected: %s', string(predictedSpecies)));
    colormap jet;
    colorbar;
end

fprintf('\n🌊 BlueFish v1.0 Execution Complete!\n');
