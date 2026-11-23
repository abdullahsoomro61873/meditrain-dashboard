# MediTrain Dashboard — City Hospital

A Live Training Dashboard for monitoring machine learning training and validation performance using Streamlit.

## Project Overview

**Client:** City Hospital  
**Project:** MediTrain Dashboard — City Hospital  
**Dataset:** Breast Cancer Wisconsin Dataset (We used a built-in dataset from Scikit-learn)

The goal is to build an interactive dashboard that allows a data team to monitor model training, visualize loss curves, diagnose model performance, and compare experiments.

**Note:** This is an educational project, not a clinically validated medical screening system.

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Dashboard Design and Workflow | Completed |
| Phase 2 | Dataset Loading and Preprocessing | Completed |
| Phase 3 | Model Training Engine | Completed |
| Phase 4 | Training Curve Diagnosis | Completed |
| Phase 5 | Streamlit Dashboard and Live Charts | Completed |
| Phase 6 | Results and Performance Metrics | Completed |
| Phase 7 | Experiments, CSV Export and Final Testing | Completed |
| Phase 8 | Final Report and Recommendation | Completed |

## Phase 1 — Dashboard Design and Workflow

### Workflow

1. Load the Breast Cancer Wisconsin dataset.
2. Split the dataset into training (60%), validation (20%), and testing (20%).
3. Configure model settings through the dashboard sidebar.
4. Train the model one epoch at a time.
5. Visualize training and validation loss live.
6. Diagnose curves as healthy, overfitting, or underfitting.
7. Display experiment results and allow CSV downloads.
8. Perform final evaluation and prepare a recommendation report.

### Planned Dashboard Components

- Sidebar for training configurations.
- Train button.
- Live training and validation loss chart.
- Training progress bar.
- Model diagnosis.
- Experiment history table.
- CSV download option.
- Educational patient-data demonstration.

## Phase 2 — Dataset Loading and Preprocessing

- Loaded Breast Cancer Wisconsin dataset (569 samples, 30 features).
- Converted labels: Malignant(cancerous tumors) = 1, Benign(non-cancerous tumors) = 0.
- Split dataset into 60% training, 20% validation and 20% testing.
- Applied StandardScaler fitted only on training data.
- Added optional scaling ON/OFF functionality.
- Verified dataset shapes and scaling results.

### Results

| Test | Expected | Actual | Status |
|---|---:|---:|---|
| Total Samples | 569 | 569 | Passed |
| Features | 30 | 30 | Passed |
| Training Samples | 341 | 341 | Passed |
| Validation Samples | 114 | 114 | Passed |
| Testing Samples | 114 | 114 | Passed |
| Training Mean | ≈ 0 | 4.86e-16 | Passed |
| Training Std | ≈ 1 | 1.0 | Passed |
| Scaling OFF | None | None | Passed |

**Conclusion:** Dataset loading, label conversion, data splitting, and optional scaling successfully verified.

## Phase 3 — Model Training Engine

- Created a PyTorch neural network for binary classification.
- Implemented configurable hidden-layer neurons.
- Added dropout regularization.
- Implemented weight decay using Adam optimizer.
- Used BCEWithLogitsLoss for binary classification.
- Calculated training and validation loss after every epoch.
- Implemented epoch callback functionality (`on_epoch`).
- Added early stopping with configurable patience.
- Saved and restored the best model state.
- Stored training and validation losses in history.

### Results

| Test | Expected | Actual | Status |
|---|---|---|---|
| Model Training | Successful | Successful | Passed |
| Epochs Run | 30 | 30 | Passed |
| Best Validation Loss | Calculated | 0.0712 | Passed |
| Callback Calls | 30 | 30 | Passed |
| Training History | 30 | 30 | Passed |
| Validation History | 30 | 30 | Passed |
| Early Stopping | Enabled | 4 epochs | Passed |
| Best Model Restoration | Successful | Successful | Passed |

**Conclusion:** The PyTorch training engine successfully trains the model, records training and validation losses, supports epoch callbacks, and implements early stopping with best model restoration.

## Phase 4 — Training Curve Diagnosis

- Created `diagnosis.py` for analyzing training and validation loss curves.
- Implemented underfitting detection (training loss > 0.3).
- Implemented overfitting detection (final validation loss 30% above its best).
- Implemented healthy model diagnosis.
- Added empty training history validation.
- Verified all three diagnosis conditions.

### Diagnosis Rules

| Condition | Diagnosis |
|---|---|
| Training loss > 0.3 | Underfitting |
| Final validation loss ≥ 1.3 × best validation loss | Overfitting |
| Neither condition applies | Healthy |

### Results

| Test | Expected | Actual | Status |
|---|---|---|---|
| Underfitting | underfitting | underfitting | Passed |
| Overfitting | overfitting | overfitting | Passed |
| Healthy | healthy | healthy | Passed |
| Empty History | ValueError | ValueError | Passed |

**Conclusion:** The diagnosis function successfully identifies underfitting, overfitting, and healthy training curves while handling empty histories safely.

## Phase 5 — Streamlit Dashboard and Live Charts

### Implementation

Created `src/app.py` to build an interactive Streamlit dashboard.

Implemented features:

- Streamlit dashboard interface.
- Sidebar training configuration.
- Adjustable epochs, learning rate, hidden layer size, dropout, and batch size.
- Data scaling ON/OFF option.
- Start Training button.
- Model training connected with `src/trainer.py`.
- Live training progress bar.
- Training and validation loss charts.
- Chart updates after every 5 epochs.
- Final training results display.

### Test Results

| Metric | Scaling ON | Scaling OFF |
|---|---:|---:|
| Epochs Requested | 30 | 30 |
| Epochs Run | 30 | 29 |
| Best Validation Loss | 0.4386 | 0.2030 |
| Training | Passed | Passed |
| Progress Bar | Passed | Passed |
| Live Loss Charts | Passed | Passed |

### Conclusion

The Streamlit dashboard was successfully implemented and tested with data scaling enabled and disabled. Model training, progress tracking, and live loss chart updates worked successfully.

**Note:** This project is an educational machine learning experiment. It is not clinically validated and must not be used for real patient diagnosis or medical decisions.

## Phase 6 — Model Evaluation and Results

### Implementation

Created `src/evaluation.py` to evaluate the trained model using the held-out test dataset.

Implemented metrics:

- Test Accuracy
- Cancers Found (True Positives)
- False Alarms (False Positives)
- True Negatives
- Missed Cancers (False Negatives)

Connected model diagnosis and evaluation results to the Streamlit dashboard.

### Test Configuration

| Setting | Value |
|---|---:|
| Epochs | 30 |
| Learning Rate | 0.001 |
| Hidden Layer Size | 16 |
| Dropout | 0.3 |
| Batch Size | 32 |

### Evaluation Results

| Metric | Actual Result |
|---|---:|
| Test Samples | 114 |
| Test Accuracy | 95.61% |
| Cancers Found | 39 |
| False Alarms | 2 |
| True Negatives | 70 |
| Missed Cancers | 3 |

### Verification

- Model evaluation function tested successfully.
- All 114 test samples accounted for.
- Training curve diagnosis displayed on dashboard.
- Evaluation metrics displayed on dashboard.

### Conclusion

Model evaluation and dashboard integration were successfully implemented and tested.

**Important:** These results are from an educational dataset experiment. The model is not clinically validated and must not be used for real patient diagnosis or medical decisions.

## Phase 7 — Experiment Tracking, Sample Demonstration and Final Decision

### 1. Experiment Tracking

An experiment tracking system was implemented to record and compare model training experiments.

Features:
- Records each experiment's configuration.
- Stores training and validation loss.
- Displays experiment history in a table.
- Provides CSV download functionality.

### 2. Experiment Configuration

Four experiments were conducted by changing one hyperparameter at a time.

| Experiment | Epochs | Learning Rate | Hidden Size | Dropout | Batch Size | Scaling |
|---|---:|---:|---:|---:|---:|---|
| 1 | 30 | 0.0001 | 16 | 0.3 | 32 | ON |
| 2 | 30 | 0.005 | 16 | 0.3 | 32 | ON |
| 3 | 30 | 0.0001 | 32 | 0.3 | 32 | ON |
| 4 | 30 | 0.0001 | 16 | 0.5 | 32 | ON |

### 3. Experiment Results

| Experiment | Best Validation Loss | Final Training Loss | Final Validation Loss | Diagnosis |
|---|---:|---:|---:|---|
| 1 | 0.4386 | 0.4781 | 0.4386 | Underfitting |
| 2 | 0.0414 | 0.0458 | 0.0414 | Healthy |
| 3 | 0.3929 | 0.4356 | 0.3929 | Underfitting |
| 4 | 0.4486 | 0.4966 | 0.4486 | Underfitting |

### 4. Best Model Selection

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

**Selection Reason:**

Experiment 2 achieved the lowest validation loss among all four experiments and received a healthy training diagnosis.

The model was selected using validation results, without using the held-out test dataset for experiment tuning.

### 5. Educational Sample Demonstration

An educational sample prediction feature was added to the dashboard.

#### How to Use

1. Start model training.
2. Wait until training is completed.
3. Navigate to Educational Sample Demonstration.
4. Select an example sample from the dropdown.
5. Click the **Predict Example Sample** button.
6. Review the prediction results.

#### Prediction Outputs

- **Predicted Class:** Model prediction (Malignant or Benign).
- **Predicted Probability Score:** Model output probability score.
- **Dataset Label:** Actual category from the dataset.

#### Demonstration Rules and Limitations

1. Model training must be completed before using this feature.
2. Only validation dataset examples are used.
3. The feature is for educational demonstration only.
4. Matching a dataset label does not prove medical reliability.
5. The probability score is not a clinically validated risk estimate.
6. Real patient information must not be entered or used.
7. The model must not be used for real patient diagnosis or medical decisions.

### 6. Final Test Results

The held-out test evaluation was performed in Phase 6.

Previously recorded results:

| Metric | Result |
|---|---:|
| Test Accuracy | 95.61% |
| Cancers Found (TP) | 39 |
| False Alarms (FP) | 2 |
| True Negatives (TN) | 70 |
| Missed Cancers (FN) | 3 |
| Total Test Samples | 114 |

The test results are documented from the Phase 6 evaluation. The test dataset was not reused for Phase 7 experiment tuning.

**Important:** These test results belong to the model configuration evaluated in Phase 6; they should not automatically be attributed to Experiment 2.

### 7. Final Decision

**Selected Model: Experiment 2**

Experiment 2 was selected based on its validation performance.

However, the project is an educational machine learning demonstration and is not ready for real-world clinical use.

#### Three Reasons the Model Is Not Ready for Real-World Use

1. **Limited Dataset:** The model uses a single public dataset with a limited number of samples. Its performance may not generalize to other populations or data sources.

2. **No Clinical Validation:** The model has not undergone clinical testing, external validation, or approval by qualified medical professionals.

3. **Prediction Errors:** The Phase 6 test evaluation recorded 2 false alarms and 3 missed cancers. These errors demonstrate that the model cannot be relied upon for medical decisions.

### 8. Phase 7 Deliverables

- Experiment tracking table implemented.
- CSV download functionality implemented.
- Four hyperparameter experiments completed.
- Experiment results compared.
- Best model selected using validation results.
- Educational sample demonstration implemented and tested.
- Final decision documented.
- Project limitations and safety disclaimer documented.

## Project Structure

```text
meditrain-dashboard/
│
├── .venv/
├── data/
├── results/
├── src/
|    └── data.py
|    └── trainer.py
|    └── diagnosis.py
|    └── app.py
|    └── evaluation.py
├── README.md
|── FINAL_REPORT.md
├── requirements.txt
└── .gitignore
```
## Installation and Setup

The following steps describe how to set up and run the project on Windows using PowerShell.

### 1. Clone the GitHub Repository

```powershell
git clone https://github.com/abdullahsoomro61873/meditrain-dashboard
cd meditrain-dashboard
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Phase 2 — Dataset Testing

**Test dataset splitting:**

```powershell
python -c "from src.data import load_data; d = load_data(); print('Train:', d[0].shape); print('Validation:', d[1].shape); print('Test:', d[2].shape)"
```

**Test scaling:**

```powershell
python -c "from src.data import load_data; import numpy as np; d = load_data(); print('Training mean:', np.mean(d[0])); print('Training std:', np.std(d[0]))"
```

**Test scaling OFF:**

```powershell
python -c "from src.data import load_data; d = load_data(scale=False); print('Train:', d[0].shape); print('Scaler:', d[6])"
```

### 6. Phase 3 — Training Engine Testing

**Test model training and epoch callbacks:**

```powershell
python -c "from src.data import load_data; from src.trainer import train; events=[]; model,hist=train({'hidden_size':16,'dropout':0.3,'learning_rate':0.001,'weight_decay':0.01,'epochs':30,'batch_size':32,'patience':8,'early_stopping':True}, load_data(), on_epoch=lambda epoch,h: events.append(epoch)); print('Epochs run:',hist['epochs_run']); print('Best validation loss:',round(hist['best_val_loss'],4)); print('Callback calls:',len(events)); print('Training history:',len(hist['train'])); print('Validation history:',len(hist['val']))"
```

**Test early stopping:**

```powershell
python -c "from src.data import load_data; from src.trainer import train; cfg={'hidden_size':16,'dropout':0.3,'learning_rate':0,'epochs':20,'patience':3,'early_stopping':True}; model,hist=train(cfg,load_data()); print('Requested epochs: 20'); print('Actual epochs:',hist['epochs_run']); print('Early stopping:',hist['epochs_run'] < 20)"
```

Note: The zero learning rate in the early-stopping test is intentional and used only to verify the stopping mechanism.

### 7. Phase 4 — Training Curve Diagnosis

Run the following commands to verify underfitting, overfitting, healthy model diagnosis, and empty history validation.

**Test all three diagnosis conditions:**

```powershell
python -c "from src.diagnosis import diagnose; tests={'Underfitting':{'train':[0.8,0.6,0.5],'val':[0.7,0.6,0.5]},'Overfitting':{'train':[0.2,0.1,0.05],'val':[0.2,0.1,0.15]},'Healthy':{'train':[0.2,0.15,0.1],'val':[0.25,0.18,0.12]}}; [print(name, '->', diagnose(hist)) for name,hist in tests.items()]"
```

**Test empty training history:**

```powershell
python -c "from src.diagnosis import diagnose; diagnose({'train': [], 'val': []})"
```

The second command is expected to raise `ValueError: Training history cannot be empty.`

### 8. Run the Dashboard

```powershell
streamlit run src/app.py
```

The dashboard will be available after the Streamlit application is implemented.

### 9. Phase 6 — Model Evaluation

Run the evaluation test from the project root directory:

```powershell
python -c "from src.data import load_data; from src.trainer import train; from src.evaluation import evaluate_model; data=load_data(); config={'epochs':30,'learning_rate':0.001,'hidden_size':16,'dropout':0.3,'batch_size':32,'patience':20,'early_stopping':True,'seed':42}; model,hist=train(config,data); results=evaluate_model(model,data); print('Test Accuracy:',round(results['test_accuracy'],4)); print('Cancers Found:',results['cancers_found']); print('False Alarms:',results['false_alarms']); print('True Negatives:',results['true_negatives']); print('Missed Cancers:',results['missed_cancers'])"
```

Run the Streamlit dashboard:

```powershell
streamlit run src/app.py
```

Verify the training diagnosis and evaluation metrics on the dashboard.

**Note:** The held-out test dataset is reserved for final evaluation. Do not repeatedly use it for model tuning.

## Technologies

- Python
- PyTorch
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
