
def diagnose(hist):

    train_losses = hist["train"]
    val_losses = hist["val"]

    # Check empty history
    if not train_losses or not val_losses:
        raise ValueError("Training history cannot be empty.")

    # Final training loss
    final_train_loss = train_losses[-1]

    # Best and final validation loss
    best_val_loss = min(val_losses)
    final_val_loss = val_losses[-1]

    # Underfitting
    if final_train_loss > 0.3:
        return "underfitting"

    # Overfitting
    if final_val_loss >= best_val_loss * 1.3:
        return "overfitting"

    # Healthy model
    return "healthy"
