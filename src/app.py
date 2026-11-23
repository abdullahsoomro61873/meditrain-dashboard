
# ==========================================
# MediTrain - Live Training Dashboard
# Phase 7: Experiment Tracking
# ==========================================

import streamlit as st
import pandas as pd
import torch

from datetime import datetime

# Import project modules
from data import load_data
from trainer import train
from diagnosis import diagnose

# ------------------------------------------
# 1. Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="MediTrain Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediTrain — Live Training Dashboard")

st.write(
    "Educational dashboard for machine learning "
    "training and experiment tracking."
)

# Initialize experiment storage for this session
if "experiments" not in st.session_state:
    st.session_state.experiments = []


# ------------------------------------------
# 2. Sidebar Training Settings
# ------------------------------------------

st.sidebar.header("Training Settings")

epochs = st.sidebar.slider(
    "Number of Epochs",
    10, 200, 100, 10
)

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

hidden_size = st.sidebar.selectbox(
    "Hidden Layer Size",
    [8, 16, 32, 64],
    index=1
)

dropout = st.sidebar.slider(
    "Dropout",
    0.0, 0.8, 0.3, 0.1
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64],
    index=1
)

scale_data = st.sidebar.checkbox(
    "Enable Data Scaling",
    value=True
)


# ------------------------------------------
# 3. Display Configuration
# ------------------------------------------

st.subheader("Selected Training Configuration")

st.write({
    "Epochs": epochs,
    "Learning Rate": learning_rate,
    "Hidden Layer Size": hidden_size,
    "Dropout": dropout,
    "Batch Size": batch_size,
    "Data Scaling": scale_data
})

st.divider()


# ------------------------------------------
# 4. Start Model Training
# ------------------------------------------

st.subheader("Model Training")

if st.button("Start Training", type="primary"):

    # Load dataset with selected scaling option
    data = load_data(scale=scale_data)

    # Prepare training configuration
    config = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "batch_size": batch_size,
        "patience": 20,
        "early_stopping": True,
        "seed": 42
    }

    # Create dashboard placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()

    # Callback for live training updates
    def update_training_chart(epoch, history):

        progress = min(epoch / epochs, 1.0)

        progress_bar.progress(progress)

        status_text.write(
            f"Training: Epoch {epoch}/{epochs}"
        )

        # Update chart after every 5 epochs
        if epoch % 5 == 0:

            chart_data = pd.DataFrame({
                "Training Loss": history["train"],
                "Validation Loss": history["val"]
            })

            chart_placeholder.line_chart(chart_data)

    # Train model
    model, history = train(
        config,
        data,
        on_epoch=update_training_chart
    )

    # Save model and validation samples for demo
    st.session_state.demo_model = model
    st.session_state.demo_X_val = data[1]
    st.session_state.demo_y_val = data[4]

    # Display final loss chart
    final_chart = pd.DataFrame({
        "Training Loss": history["train"],
        "Validation Loss": history["val"]
    })

    chart_placeholder.line_chart(final_chart)

    progress_bar.progress(1.0)

    status_text.success("Training completed!")

    # --------------------------------------
    # 5. Diagnose Training Curves
    # --------------------------------------

    model_diagnosis = diagnose(history)

    st.subheader("Training Curve Diagnosis")

    st.write("Diagnosis:", model_diagnosis.title())

    # --------------------------------------
    # 6. Record Experiment
    # --------------------------------------

    experiment = {
        "Experiment": len(st.session_state.experiments) + 1,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Epochs Requested": epochs,
        "Epochs Run": history["epochs_run"],
        "Learning Rate": learning_rate,
        "Hidden Size": hidden_size,
        "Dropout": dropout,
        "Batch Size": batch_size,
        "Data Scaling": scale_data,
        "Best Validation Loss": round(
            history["best_val_loss"], 4
        ),
        "Final Training Loss": round(
            history["train"][-1], 4
        ),
        "Final Validation Loss": round(
            history["val"][-1], 4
        ),
        "Diagnosis": model_diagnosis
    }

    st.session_state.experiments.append(experiment)

    st.success("Experiment saved successfully!")

    # Show current experiment results
    st.subheader("Latest Experiment")

    st.write(experiment)

    st.info(
        "Experiment comparisons use training and validation "
        "results. The held-out test dataset is not evaluated "
        "during these experiment runs."
    )

# ------------------------------------------
# 7. Educational Sample Demonstration
# ------------------------------------------

st.divider()

st.subheader("Educational Sample Demonstration")

st.warning(
    "This demonstration uses an example from the "
    "validation dataset. It is not a real patient "
    "assessment or medical diagnosis."
)

if "demo_model" in st.session_state:

    X_val = st.session_state.demo_X_val
    y_val = st.session_state.demo_y_val
    model = st.session_state.demo_model

    sample_index = st.selectbox(
        "Select an Example Sample",
        range(len(X_val))
    )

    if st.button("Predict Example Sample"):

        sample = torch.tensor(
            X_val[sample_index],
            dtype=torch.float32
        ).reshape(1, -1)

        model.eval()

        with torch.no_grad():
            logits = model(sample)
            probability = torch.sigmoid(logits).item()

        prediction = (
            "Malignant" if probability >= 0.5
            else "Benign"
        )

        actual = (
            "Malignant" if int(y_val[sample_index]) == 1
            else "Benign"
        )

        st.write("**Predicted Class:**", prediction)

        st.write(
            "**Predicted Probability Score:**",
            round(probability, 4)
        )

        st.write("**Dataset Label:**", actual)

else:

    st.info(
        "Please start model training first "
        "to enable the sample demonstration."
    )
# ------------------------------------------
# 8. Experiments Table
# ------------------------------------------

st.divider()

st.subheader("Experiment History")

if st.session_state.experiments:

    experiments_df = pd.DataFrame(
        st.session_state.experiments
    )

    # Display all recorded experiments
    st.dataframe(
        experiments_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------
    # 9. Download Experiments as CSV
    # --------------------------------------

    csv_data = experiments_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Experiments CSV",
        data=csv_data,
        file_name="meditrain_experiments.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No experiments recorded yet. "
        "Start training to create your first experiment."
    )


# ------------------------------------------
# 10. Educational Disclaimer
# ------------------------------------------

st.divider()

st.info(
    "This project is for educational machine learning "
    "experiments only. It is not clinically validated "
    "and must not be used for real patient diagnosis "
    "or medical decisions."
)
