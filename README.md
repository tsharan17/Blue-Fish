# 🌊 BlueFish v2.0

BlueFish is a MATLAB workflow for marine species identification from underwater audio. Version 2.0 bundles both the paper-ready 10-fold cross-validation experiment and an interactive demo that classifies a single `.wav` file with the project’s best-performing model.

## Key Features
- Dual operation modes:
  - **Experiment mode** reproduces benchmark metrics for publications via 10-fold CV.
  - **Demo mode** trains on all cleaned data and performs single-file identification.
- Rich statistical feature extraction using MFCC mean, standard deviation, skewness, and kurtosis.
- Automatic dataset cleaning with per-class minimum sample thresholds.
- Random Forest, ECOC (Gaussian SVM), and kNN baselines for benchmarking.
- Confusion matrix visualization, spectrogram display, and audio playback for demos.

## Prerequisites
- MATLAB R2021a or newer recommended.
- Signal Processing Toolbox and Statistics and Machine Learning Toolbox.
- A dataset organized as `<dataset root>/<species name>/*.wav`, with mono or stereo files (stereo collapses to mono automatically).

## Getting Started
1. Clone or download this repository.
2. Open MATLAB and set the working directory to the project folder.
3. Place or link your dataset folder; ensure each species lives in its own subfolder.
4. Run `BlueFish_v2.m`. MATLAB will prompt for mode selection and dataset location via GUI dialogs.

## Usage

### Experiment Mode
Use **“Run Experiment (for Paper)”** to:
1. Load and clean the dataset.
2. Extract MFCC-based statistical features.
3. Perform 10-fold cross-validation with ECOC, kNN, and Random Forest.
4. Output mean accuracy and standard deviation for each classifier, plus a confusion matrix for Random Forest.

Copy the printed metrics directly into publication tables.

### Demo Mode
Use **“Run Demo (Test Single File)”** to:
1. Train the Random Forest classifier on all cleaned samples.
2. Save the trained model to `bluefish_demo_model_rf.mat`.
3. Select any `.wav` file to classify.
4. View the predicted species, inspect a spectrogram, and hear the audio playback.

## Tips & Troubleshooting
- Increase `minSamples` in the script if you want stricter class inclusion; reduce it for small datasets.
- Ensure audio files are not clipped—pre-normalization is handled, but extremely noisy signals can degrade accuracy.
- If the GUI dialogs fail inside headless environments, replace `uigetdir`/`uigetfile` with hard-coded paths.

## Extending BlueFish
- Swap in alternative classifiers by editing the anonymous functions in the experiment section.
- Add hyperparameter searches or feature engineering for new research directions.
- Export trained models to MATLAB apps or deploy to embedded hardware.

## Citation
If this workflow supports your research, please cite the corresponding paper or this repository to acknowledge the tool.

---

Happy exploring beneath the waves! 🌊