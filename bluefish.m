% =========================================================================
%           INTELLIGENT ACOUSTIC PROCESSING FOR MARINE SPECIES MONITORING
%                             (All-in-One Script)
% =========================================================================
%
% This single script handles the entire process:
% 1. Creates the species database in memory.
% 2. Prompts the user to select an audio file.
% 3. Processes the audio to find dominant frequencies using a local function.
% 4. Identifies species by comparing frequencies to the database using another local function.
% 5. Displays the results in the command window.
%
% To run:
%   1. Save this entire code as a single .m file (e.g., 'marine_acoustics_analyzer.m').
%   2. Run the file in MATLAB.
%   3. Select your ocean audio file when the dialog box appears.
%
% =========================================================================
%                       --- MAIN SCRIPT LOGIC ---
% =========================================================================
clear; clc; close all;

fprintf('Marine Species Identification from Acoustic Data (All-in-One Script)\n');
fprintf('=======================================================================\n\n');

% --- Step 1: Define the Species Database in the Script ---
% No need to load a .mat file; we create the database directly.
speciesNames = {
    'Blue Whale'; 'Humpback Whale'; 'Fin Whale'; 'Bowhead Whale';
    'Bottlenose Dolphin'; 'Orca (Killer Whale)'; 'Walrus'; 'Bearded Seal'
};
minFrequencies = [10; 20; 15; 25; 200; 1000; 200; 100];
maxFrequencies = [100; 8000; 100; 900; 20000; 30000; 1600; 4000];
speciesData = table(speciesNames, minFrequencies, maxFrequencies, ...
    'VariableNames', {'Name', 'MinFreq', 'MaxFreq'});
fprintf('Species database defined internally.\n');
disp(speciesData);


% --- Step 2: User Selects an Audio File ---
[fileName, pathName] = uigetfile({'*.wav';'*.mp3';'*.flac'}, 'Select an Ocean Audio File');
if isequal(fileName, 0)
    disp('User selected Cancel. Exiting program.');
    return;
end
audioFilePath = fullfile(pathName, fileName);
fprintf('\nSelected audio file: %s\n', audioFilePath);


% --- Step 3: Process and Identify ---
try
    % Let's identify the top 5 dominant frequencies for this example
    numPeaks = 5;
    
    % Call the LOCAL function to process the audio
    dominantFrequencies = process_audio(audioFilePath, numPeaks);
    
    if isempty(dominantFrequencies)
        fprintf('No significant frequencies detected in the audio file.\n');
    else
        fprintf('\nDetected Dominant Frequencies (Hz):\n');
        disp(dominantFrequencies');
        
        % Call the LOCAL function to identify species
        identifiedSpecies = identify_species(dominantFrequencies, speciesData);
        
        % Display the final results
        fprintf('\n--- Identification Results ---\n');
        if isempty(identifiedSpecies)
            fprintf('No matching marine species found in the database for the detected frequencies.\n');
        else
            fprintf('Potential marine species identified in the audio:\n');
            for i = 1:length(identifiedSpecies)
                fprintf('- %s\n', identifiedSpecies{i});
            end
        end
    end
    
catch ME
    fprintf('\nAn error occurred: %s\n', ME.message);
    fprintf('Please check if the audio file is valid.\n');
end

fprintf('\n=======================================================================\n');
fprintf('Analysis complete.\n');


% =========================================================================
%                       --- LOCAL FUNCTIONS ---
% These functions are only visible to the script above in this same file.
% =========================================================================

function dominantFrequencies = process_audio(filePath, numPeaks)
    % Reads an audio file, performs FFT, and finds dominant frequencies.
    [y, fs] = audioread(filePath);
    if size(y, 2) > 1, y = mean(y, 2); end % Convert to mono
    
    L = length(y);
    n = 2^nextpow2(L);
    Y = fft(y, n);
    P2 = abs(Y/n);
    P1 = P2(1:n/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = fs*(0:(n/2))/n;
    
    minProminence = 0.1 * max(P1);
    [~, locs] = findpeaks(P1, 'MinPeakProminence', minProminence, 'SortStr', 'descend', 'NPeaks', numPeaks);
    
    dominantFrequencies = f(locs);
    
    % Plotting the spectrum for visualization
    figure;
    plot(f, P1);
    title('Single-Sided Amplitude Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('|Amplitude|');
    hold on;
    if ~isempty(locs)
        plot(dominantFrequencies, P1(locs), 'r*', 'MarkerSize', 8);
        legend('Spectrum', 'Detected Peaks');
    end
    xlim([0 fs/8]);
    grid on;
end


function identifiedSpecies = identify_species(frequencies, speciesData)
    % Compares detected frequencies against a species database.
    identifiedSpecies = {};
    for i = 1:length(frequencies)
        currentFreq = frequencies(i);
        for j = 1:height(speciesData)
            if currentFreq >= speciesData.MinFreq(j) && currentFreq <= speciesData.MaxFreq(j)
                speciesName = speciesData.Name{j};
                if ~ismember(speciesName, identifiedSpecies)
                    identifiedSpecies{end+1} = speciesName;
                end
            end
        end
    end
end
