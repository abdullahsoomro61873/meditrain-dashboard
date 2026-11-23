
# ==========================================
# MediTrain - Model Evaluation
# Phase 6: Model Evaluation and Results
# ==========================================

import numpy as np
import torch


def evaluate_model(model, data):

    # Get the unseen test dataset
    X_test, y_test = data[2], data[5]

    # Convert test data into PyTorch tensors
    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # Set model to evaluation mode
    model.eval()

    # Generate predictions without calculating gradients
    with torch.no_grad():
        logits = model(X_test)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).int().numpy().flatten()

    # Actual labels and predicted labels
    actual = np.asarray(y_test).astype(int)

    # Calculate confusion matrix values
    true_positives = int(
        np.sum((actual == 1) & (predictions == 1))
    )

    false_positives = int(
        np.sum((actual == 0) & (predictions == 1))
    )

    true_negatives = int(
        np.sum((actual == 0) & (predictions == 0))
    )

    false_negatives = int(
        np.sum((actual == 1) & (predictions == 0))
    )

    # Calculate test accuracy
    accuracy = float(
        np.mean(actual == predictions)
    )

    # Return evaluation results
    return {
        "test_accuracy": accuracy,
        "cancers_found": true_positives,
        "false_alarms": false_positives,
        "true_negatives": true_negatives,
        "missed_cancers": false_negatives
    }
