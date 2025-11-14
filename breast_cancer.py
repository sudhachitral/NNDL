import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(input_dim, lr=1e-3, dropout_rate=0.3):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0,1],[0,1], linestyle='--', color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

def load_wisconsin_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

def main():
    X, y = load_wisconsin_data()
    print("Dataset shape:", X.shape)
    print("Class distribution:\n", y.value_counts())
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=SEED, stratify=y_trainval)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("Class weights:", class_weight_dict)
    model = build_model(input_dim=X_train_scaled.shape[1], lr=1e-3, dropout_rate=0.25)
    model.summary()
    ckpt_path = 'best_model.h5'
    cb_early = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    cb_ckpt = callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss')
    history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                        epochs=100, batch_size=32, class_weight=class_weight_dict,
                        callbacks=[cb_early, cb_reduce, cb_ckpt], verbose=2)
    plot_history(history)
    y_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_proba)
    import joblib
    joblib.dump(scaler, 'scaler.joblib')
    model.save('final_model.h5')
    print("Saved scaler.joblib and final_model.h5")
    sample = X_test_scaled[:3]
    sample_proba = model.predict(sample).ravel()
    print("Sample predictions:", sample_proba)

if __name__ == '__main__':
    main()
