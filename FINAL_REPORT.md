# MediTrain — Final Project Report

## 1. Project Overview

MediTrain is an educational machine learning project that demonstrates model training, live training visualization, experiment tracking, model evaluation, and sample prediction using the Breast Cancer Wisconsin dataset.

The project uses a Streamlit dashboard to visualize the training process and compare model experiments.

**Purpose:** Educational machine learning demonstration only.

## 2. Dataset

- Dataset: Breast Cancer Wisconsin
- Source: Scikit-learn built-in dataset
- Total Samples: 569
- Total Features: 30
- Malignant Samples: 212
- Benign Samples: 357

### Dataset Split

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 341 | 60% |
| Validation | 114 | 20% |
| Testing | 114 | 20% |
| Total | 569 | 100% |

StandardScaler was fitted only on the training dataset and applied to validation and test datasets.

## 3. Model Architecture

The project uses a neural network classifier implemented with PyTorch.

Architecture:
- Input Layer: 30 features
- Hidden Layer: 16 neurons
- Activation: ReLU
- Dropout: 0.3
- Output Layer: 1 neuron
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam

## 4. Training Dashboard

The Streamlit dashboard provides:

- Training configuration controls.
- Start Training button.
- Live training and validation loss charts.
- Training progress bar.
- Training curve diagnosis.
- Experiment history table.
- CSV download functionality.
- Educational sample prediction.

## 5. Experiment Results

Four experiments were conducted by changing one hyperparameter at a time.

| Experiment | Learning Rate | Hidden Size | Dropout | Best Validation Loss | Diagnosis |
|---|---:|---:|---:|---:|---|
| 1 | 0.0001 | 16 | 0.3 | 0.4386 | Underfitting |
| 2 | 0.005 | 16 | 0.3 | 0.0414 | Healthy |
| 3 | 0.0001 | 32 | 0.3 | 0.3929 | Underfitting |
| 4 | 0.0001 | 16 | 0.5 | 0.4486 | Underfitting |

## 6. Best Model Selection

**Selected Model: Experiment 2**

Configuration:
- Epochs: 30
- Learning Rate: 0.005
- Hidden Size: 16
- Dropout: 0.3
- Batch Size: 32
- Data Scaling: Enabled

Results:
- Best Validation Loss: 0.0414
- Final Training Loss: 0.0458
- Final Validation Loss: 0.0414
- Diagnosis: Healthy

Experiment 2 was selected because it achieved the lowest validation loss among the four experiments.

## 7. Final Test Evaluation

The following results were obtained during Phase 6 evaluation:

| Metric | Result |
|---|---:|
| Test Accuracy | 95.61% |
| True Positives | 39 |
| False Positives | 2 |
| True Negatives | 70 |
| False Negatives | 3 |
| Total Test Samples | 114 |

These are the previously recorded Phase 6 results. They belong to the configuration evaluated during that phase and are not automatically the test results of Experiment 2.

## 8. Model Limitations

The model is not ready for real-world clinical use for the following reasons:

1. Limited Dataset: The model uses one public dataset with a limited number of samples.
2. No Clinical Validation: The model has not undergone clinical validation or approval.
3. Prediction Errors: The test evaluation recorded false alarms and missed cancer cases.

## 9. Final Recommendation

Experiment 2 is the selected model based on validation performance.

The project successfully demonstrates machine learning training, experiment comparison, evaluation, and visualization.

However, it is strictly an educational project and must not be used for real patient diagnosis or medical decisions.

## 10. Conclusion

MediTrain demonstrates an end-to-end educational machine learning workflow, from dataset preprocessing to model training, experiment tracking, evaluation, and sample prediction.

The project objectives were implemented and documented as an educational demonstration.