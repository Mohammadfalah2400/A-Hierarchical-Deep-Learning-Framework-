# 

import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(split_dir):
    """Load train, validation, and test sets from .pkl files."""
    X_train = joblib.load(os.path.join(split_dir, "X_train.pkl"))
    y_train = joblib.load(os.path.join(split_dir, "y_train.pkl"))
    X_valid = joblib.load(os.path.join(split_dir, "X_valid.pkl"))
    y_valid = joblib.load(os.path.join(split_dir, "y_valid.pkl"))
    X_test = joblib.load(os.path.join(split_dir, "X_test.pkl"))
    y_test = joblib.load(os.path.join(split_dir, "y_test.pkl"))

    # CNN expects 3D input: (samples, timesteps, channels)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def compile_model(input_shape, num_classes, architecture_fn):
    """
    Compile a CNN model using architecture_fn passed from notebook.
    architecture_fn should be a function that returns a compiled tf.keras model.
    """
    return architecture_fn(input_shape, num_classes)

# Mymodel.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def save_architecture_and_params(model, params, output_dir):
    arch_path = os.path.join(output_dir, "architecture.json")
    with open(arch_path, "w") as f:
        f.write(model.to_json())
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

def save_results(test_loss, test_acc, y_true, y_pred, label_names, output_dir):

    label_to_int = {name: idx for idx, name in enumerate(label_names)}
    if y_true.dtype.kind in {'U', 'S', 'O'}:
        y_true_idx = np.array([label_to_int[y] for y in y_true])
    else:
        y_true_idx = y_true
    if y_pred.dtype.kind in {'U', 'S', 'O'}:
        y_pred_idx = np.array([label_to_int[y] for y in y_pred])
    else:
        y_pred_idx = y_pred

    results_json = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(results_json, f, indent=4)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n")
    class_report = classification_report(
        y_true_idx, y_pred_idx, target_names=label_names, output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(class_report, f, indent=4)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true_idx, y_pred_idx, target_names=label_names))
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(label_names)))
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=True)
    with open(os.path.join(output_dir, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix (rows = true labels, columns = predicted labels):\n\n")
        f.write(cm_df.to_string())
    print("Saved confusion matrix with class names:")
    print(cm_df)


def train_and_save_model(
    model,
    X_train, y_train,
    X_valid, y_valid,
    output_dir,
    epochs=20,
    batch_size=32,
    callbacks=None,
    verbose=1
):
    """
    Train the given model and save the best model and training history.
    Training parameters (epochs, batch_size, callbacks, verbose) can be set from the notebook.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.h5")

    # If no callbacks are given, set a default ModelCheckpoint
    if callbacks is None:
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
        callbacks = [checkpoint]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    # Save training history
    history_path = os.path.join(output_dir, "history.npy")
    np.save(history_path, history.history)
    return history

def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the model on the test set and save results to a text file.
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results_path = os.path.join(output_dir, "evaluation.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
    print(f"✅ Model evaluation saved to {results_path}")
    return test_loss, test_acc
# ------------------------------------------------------------------------------------------------------------------------------------------/...........................

import pickle
import os
import tensorflow as tf
from tensorflow.keras import layers, models

def load_subgroup_data(data_dir):
    """Loads train/valid/test splits from the given directory."""
    def load_pkl(filename):
        with open(os.path.join(data_dir, filename), "rb") as f:
            return pickle.load(f)
    X_train = load_pkl("X_train.pkl")
    y_train = load_pkl("y_train.pkl")
    X_valid = load_pkl("X_valid.pkl")
    y_valid = load_pkl("y_valid.pkl")
    X_test = load_pkl("X_test.pkl")
    y_test = load_pkl("y_test.pkl")
    return (X_train, y_train, X_valid, y_valid, X_test, y_test)

from tensorflow.keras import models, layers

def build_cnn_model(input_shape, num_classes, layer_sizes=[128, 64], activation="relu"):
    """
    Builds a dense (MLP) neural network for multiclass classification with one-hot encoded labels.
    - input_shape: tuple, e.g. (n_features,)
    - num_classes: int, number of output classes (should match y.shape[1])
    - layer_sizes: list of int, size of each Dense hidden layer
    - activation: activation function for hidden layers
    Returns:
        Compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for size in layer_sizes:
        model.add(layers.Dense(size, activation=activation))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",  # For one-hot encoded labels
        metrics=["accuracy"]
    )
    return model

def train_cnn_model(model, X_train, y_train, X_valid=None, y_valid=None, epochs=20, batch_size=32):
    """
    Trains the given Keras model on the provided data.
    Returns:
        History object from model.fit
    """
    if X_valid is not None and y_valid is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    return history

