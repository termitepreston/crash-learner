from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight


def get_class_weights(y_train):
    """
    Computes class weights to handle dataset imbalance.
    Useful because 'Minor' accidents dominate the dataset.
    """
    # Ensure y_train is a 1D array
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.flatten()

    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )

    # Convert to dictionary format required by Keras {class_index: weight}
    weight_dict = dict(zip(classes, weights))
    print(f"[INFO] Class Weights computed: {weight_dict}")
    return weight_dict


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32,
    epochs=100,
    model_save_path="models/best_model.keras",
):
    """
    Compiles and trains the Keras model with callbacks for overfitting mitigation.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        batch_size (int): Samples per gradient update.
        epochs (int): Max number of epochs.
        model_save_path (str): Path to save the best model artifact.

    Returns:
        history: Keras History object containing loss/accuracy logs.
    """

    # 1. Compute Class Weights (Mitigation for Imbalance)
    class_weights = get_class_weights(y_train)

    # 2. Define Callbacks (Mitigation for Overfitting)
    callbacks = [
        # Stop training if validation loss doesn't improve for 15 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        # Save the model with the best validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce learning rate if loss plateaus (Optimization)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    # 3. Train the Model
    print(
        f"[INFO] Starting training for {epochs} epochs with batch size {batch_size}..."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,  # Crucial for Imbalanced Data
        callbacks=callbacks,
        verbose=1,
    )

    return history
