import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint

import joblib

def load_kfold_data(data_dir):
    """
    Loads preprocessed data (X_array, y_array, label encoder) from a given directory.
    
    Parameters:
        data_dir (str): Path to the folder where .pkl files are stored.
    
    Returns:
        X_array (np.ndarray): Feature array, reshaped for CNN input.
        y_array (np.ndarray): Encoded label array.
        le (LabelEncoder): Trained label encoder.
    """
    X_array = joblib.load(os.path.join(data_dir, "X_array.pkl"))
    y_array = joblib.load(os.path.join(data_dir, "y_array.pkl"))
    le = joblib.load(os.path.join(data_dir, "label_encoder.pkl"))
    return X_array, y_array, le


# Compile model using architecture function
def compile_model(input_shape, num_classes, architecture_fn):
    return architecture_fn(input_shape, num_classes)

# Save architecture and parameters to JSON
def save_architecture_and_params(model, params, output_dir):
    arch_path = os.path.join(output_dir, "architecture.json")
    with open(arch_path, "w") as f:
        f.write(model.to_json())
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

# Save evaluation results and confusion matrix
def save_results(test_loss, test_acc, y_true, y_pred, label_names, output_dir):
    label_to_int = {name: idx for idx, name in enumerate(label_names)}
    y_true_idx = np.array([label_to_int[y] if y in label_to_int else y for y in y_true])
    y_pred_idx = np.array([label_to_int[y] if y in label_to_int else y for y in y_pred])

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

# Train the model and save results
def train_and_save_model(model, X_train, y_train, X_valid=None, y_valid=None,
                         output_dir=".", epochs=20, batch_size=32,
                         callbacks=None, verbose=1):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.h5")

    if callbacks is None:
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
        callbacks = [checkpoint]

    if X_valid is None or y_valid is None:
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

    np.save(os.path.join(output_dir, "history.npy"), history.history)
    return history

# Evaluate and run K-Fold training for multiple architectures
def run_kfold_multiple_architectures(X_array, y_array, label_encoder, architectures_dict,
                                     n_splits=5, epochs=50, batch_size=128, output_root="kfold_multi"):
    os.makedirs(output_root, exist_ok=True)

    for arch_name, arch_fn in architectures_dict.items():
        print(f"\n================== Training architecture: {arch_name} ==================")
        arch_dir = os.path.join(output_root, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_array, y_array)):
            print(f"\n🔁 Fold {fold + 1}/{n_splits} — {arch_name}")
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]

            model = compile_model(
                input_shape=X_train.shape[1:],
                num_classes=len(label_encoder.classes_),
                architecture_fn=arch_fn
            )

            fold_dir = os.path.join(arch_dir, f"fold_{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            save_architecture_and_params(model, {"architecture": arch_name, "fold": fold + 1}, fold_dir)

            checkpoint = ModelCheckpoint(
                filepath=os.path.join(fold_dir, "best_model.h5"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=0
            )

            history = train_and_save_model(
                model, X_train, y_train,
                X_valid=None, y_valid=None,
                output_dir=fold_dir,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint],
                verbose=1
            )

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test).argmax(axis=1)

            save_results(test_loss, test_acc, y_test, y_pred, label_encoder.classes_, fold_dir)
            accuracies.append(test_acc)
            print(f"✅ Fold {fold + 1} Accuracy: {test_acc:.4f}")

        with open(os.path.join(arch_dir, "summary.txt"), "w") as f:
            f.write(f"Average Accuracy for {arch_name} over {n_splits} folds: {np.mean(accuracies):.4f}\n")
        print(f"\n✅ {arch_name} finished. Mean Accuracy: {np.mean(accuracies):.4f}")
