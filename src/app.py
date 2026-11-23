
# ==========================================
# MediTrain - Live Training Dashboard
# Phase 5: Streamlit Dashboard and Live Charts
# ==========================================

import streamlit as st
import pandas as pd

from data import load_data
from trainer import train

# ------------------------------------------
# 1. Streamlit Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="MediTrain Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Dashboard title and description
st.title("🏥 MediTrain — Live Training Dashboard")

st.write(
    "An educational dashboard for experimenting "
    "with machine learning model training."
)

st.divider()


# ------------------------------------------
# 2. Sidebar - Training Settings
# ------------------------------------------

st.sidebar.header("⚙️ Training Settings")

# Number of training epochs
epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=10,
    max_value=200,
    value=100,
    step=10
)

# Learning rate controls optimizer updates
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

# Number of neurons in the hidden layer
hidden_size = st.sidebar.selectbox(
    "Hidden Layer Size",
    options=[8, 16, 32, 64],
    index=1
)

# Dropout helps regularize the model
dropout = st.sidebar.slider(
    "Dropout",
    min_value=0.0,
    max_value=0.8,
    value=0.3,
    step=0.1
)

# Number of samples processed in each batch
batch_size = st.sidebar.selectbox(
    "Batch Size",
    options=[16, 32, 64],
    index=1
)

# Enable or disable feature scaling
scale_data = st.sidebar.checkbox(
    "Enable Data Scaling",
    value=True
)


# ------------------------------------------
# 3. Display Selected Configuration
# ------------------------------------------

st.subheader("Selected Training Configuration")

config_display = {
    "Epochs": epochs,
    "Learning Rate": learning_rate,
    "Hidden Layer Size": hidden_size,
    "Dropout": dropout,
    "Batch Size": batch_size,
    "Data Scaling": scale_data
}

st.write(config_display)

st.divider()


# ------------------------------------------
# 4. Training Section
# ------------------------------------------

st.subheader("🚀 Model Training")

# Start training when the button is clicked
if st.button("Start Training", type="primary"):

    # Load and preprocess the dataset
    data = load_data(scale=scale_data)

    # Collect all model training settings
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

    # --------------------------------------
    # 5. Progress Bar and Chart Placeholders
    # --------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    chart_placeholder = st.empty()

    # --------------------------------------
    # 6. Live Training Callback
    # --------------------------------------

    def update_training_chart(epoch, history):

        # Calculate training progress
        progress = min(epoch / epochs, 1.0)

        progress_bar.progress(progress)

        # Display current epoch
        status_text.write(
            f"Training in progress: Epoch {epoch}/{epochs}"
        )

        # Update chart after every 5 epochs
        if epoch % 5 == 0:

            chart_data = pd.DataFrame({
                "Training Loss": history["train"],
                "Validation Loss": history["val"]
            })

            chart_placeholder.line_chart(chart_data)

    # --------------------------------------
    # 7. Train the Model
    # --------------------------------------

    model, history = train(
        config,
        data,
        on_epoch=update_training_chart
    )

    # --------------------------------------
    # 8. Display Final Training Chart
    # --------------------------------------

    final_chart = pd.DataFrame({
        "Training Loss": history["train"],
        "Validation Loss": history["val"]
    })

    chart_placeholder.line_chart(final_chart)

    # Mark progress as complete
    progress_bar.progress(1.0)

    status_text.success("Training completed successfully!")

    st.divider()

    # --------------------------------------
    # 9. Display Training Results
    # --------------------------------------

    st.subheader("📊 Training Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Epochs Run",
            history["epochs_run"]
        )

    with col2:
        st.metric(
            "Best Validation Loss",
            f"{history['best_val_loss']:.4f}"
        )

    # --------------------------------------
    # 10. Educational Disclaimer
    # --------------------------------------

    st.info(
        "This dashboard is for educational machine learning "
        "experiments only. It is not clinically validated "
        "and must not be used for real patient diagnosis "
        "or medical decisions."
    )
