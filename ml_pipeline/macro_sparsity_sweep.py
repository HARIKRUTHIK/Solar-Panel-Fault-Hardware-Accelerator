"""
FYP: CNN-Based Hardware Accelerator for Automated Solar Panel Condition and Fault Classification
Phase: Macro-Sparsity Breakdown & Collapse Analysis (1e-2 to 1e-1)

Objective:
Progressively sweep from 1e-2 (~10% sparsity) to 1e-1 (~70% sparsity) on the VALIDATION dataset
to locate the catastrophic collapse point of the network. This uncovers the complete accuracy-sparsity
phase transition curve for the FYP thesis.

Evaluated Configurations (strictly independent from pristine V5 Gold weights):
- Baseline (Untouched V5 Gold)
- 1e-3 (0.001) [Selected FYP Frontier: 1.84% sparsity]
- 1e-2 (0.010) [~9.88% sparsity: 107,808 zeros]
- 2e-2 (0.020) [Intermediate point in high-density region]
- 3e-2 (0.030) [Intermediate point in high-density region]
- 5e-2 (0.050) [Intermediate point in high-density region]
- 7e-2 (0.070) [Intermediate point in high-density region]
- 1e-1 (0.100) [~69.92% sparsity: 762,935 zeros]

OUTPUT ARTIFACTS:
1. CSV Table:   /kaggle/working/V5_MACRO_SPARSITY/V5_MACRO_SPARSITY_RESULTS.csv
2. Thesis Plot: /kaggle/working/V5_MACRO_SPARSITY/V5_MACRO_ACCURACY_COLLAPSE_CURVE.png
"""

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
MODEL_PATH = "/kaggle/input/notebooks/harikruthiksk/teacher-student/best_MobileNetV3Small_KD_CORRECTED.keras"
DATASET_ROOT = "/kaggle/input/datasets/muhaanadharbi/solar-panel-images-4-classes-final/GP-dataset-clean"
VAL_DIR = os.path.join(DATASET_ROOT, "val")

OUTPUT_DIR = "/kaggle/working/V5_MACRO_SPARSITY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EXPECTED_TOTAL_WEIGHTS = 1091188

THRESHOLDS = [
    ("Baseline (V5 Gold)", None),
    ("1e-3 (Selected Sweet Spot)", 0.001),
    ("1e-2 (0.01)",                0.010),
    ("2e-2 (0.02)",                0.020),
    ("3e-2 (0.03)",                0.030),
    ("5e-2 (0.05)",                0.050),
    ("7e-2 (0.07)",                0.070),
    ("1e-1 (0.10)",                0.100),
]

print("=" * 95)
print("FYP MACRO-SPARSITY BREAKDOWN & COLLAPSE ANALYSIS (1e-2 to 1e-1)")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Checkpoint:   {MODEL_PATH}")
print(f"Validation Path:    {VAL_DIR}")
print(f"Output Directory:   {OUTPUT_DIR}")
print(f"Total Sweep Points: {len(THRESHOLDS)}")
print("=" * 95)

# ==============================================================================
# 2. LOAD VALIDATION DATASET
# ==============================================================================
print("\n[STEP 1/4] Loading Validation Dataset...")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Validation directory not found at: {VAL_DIR}")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names
print(f"Classes ({len(class_names)}): {class_names}")

print("Extracting true ground-truth labels...")
y_true_list = []
for _, labels in val_ds:
    y_true_list.append(np.argmax(labels.numpy(), axis=1))
y_true = np.concatenate(y_true_list, axis=0)
print(f"Total validation samples: {len(y_true):,}")

val_ds_eval = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==============================================================================
# 3. LOAD PRISTINE V5 GOLD REFERENCE MODEL
# ==============================================================================
print("\n[STEP 2/4] Loading Pristine V5 Gold Model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

pristine_weights = [w.copy() for w in model.get_weights()]
total_weights_count = sum(w.size for w in pristine_weights)
print(f"Model loaded. Total weight parameters: {total_weights_count:,}")

# ==============================================================================
# 4. RUN MACRO-SPARSITY EVALUATION SWEEP
# ==============================================================================
print("\n[STEP 3/4] Running Macro-Sparsity Sweep...")

results = []
baseline_val_acc = None

for idx, (name, tau) in enumerate(THRESHOLDS, 1):
    # Always reset model to pristine weights
    model.set_weights(pristine_weights)
    
    if tau is None:
        zero_count = sum(int(np.sum(w == 0.0)) for w in pristine_weights)
        pct_zeroed = (zero_count / total_weights_count) * 100.0
        applied_weights = pristine_weights
    else:
        zero_count = 0
        applied_weights = []
        for w in pristine_weights:
            zero_mask = np.abs(w) <= tau
            zero_count += int(np.sum(zero_mask))
            w_zeroed = np.where(zero_mask, 0.0, w)
            applied_weights.append(w_zeroed)
        
        pct_zeroed = (zero_count / total_weights_count) * 100.0
        model.set_weights(applied_weights)
    
    # Predict
    y_prob = model.predict(val_ds_eval, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    if baseline_val_acc is None:
        baseline_val_acc = acc
        acc_change = 0.0
    else:
        acc_change = (acc - baseline_val_acc) * 100.0
        
    print(f"[{idx}/{len(THRESHOLDS)}] {name:28s} | Zeros: {zero_count:7d} ({pct_zeroed:6.2f}%) | Val Acc: {acc*100:6.2f}% (Δ: {acc_change:+6.2f}%) | F1: {f1_weighted:.4f}")
    
    results.append({
        "Configuration": name,
        "Threshold": tau if tau is not None else 0.0,
        "Zeroed_Weights": zero_count,
        "Pct_Zeroed": pct_zeroed,
        "Val_Accuracy": acc * 100.0,
        "Accuracy_Change_pp": acc_change,
        "Val_Precision_Weighted": prec_weighted,
        "Val_Recall_Weighted": rec_weighted,
        "Val_F1_Weighted": f1_weighted
    })
    
    del y_prob, y_pred
    gc.collect()

# Final safety restoration
model.set_weights(pristine_weights)

# ==============================================================================
# 5. COMPILE TABLE & GENERATE THESIS PLOT
# ==============================================================================
print("\n[STEP 4/4] Compiling Results and Generating Collapse Curve Plot...")
results_df = pd.DataFrame(results)

csv_path = os.path.join(OUTPUT_DIR, "V5_MACRO_SPARSITY_RESULTS.csv")
results_df.to_csv(csv_path, index=False)
print(f"✓ Saved macro-sparsity results to: {csv_path}")

display_df = results_df[[
    "Configuration",
    "Threshold",
    "Zeroed_Weights",
    "Pct_Zeroed",
    "Val_Accuracy",
    "Accuracy_Change_pp",
    "Val_F1_Weighted"
]].copy()

display_df["Pct_Zeroed"] = display_df["Pct_Zeroed"].apply(lambda x: f"{x:.2f}%")
display_df["Val_Accuracy"] = display_df["Val_Accuracy"].apply(lambda x: f"{x:.2f}%")
display_df["Accuracy_Change_pp"] = display_df["Accuracy_Change_pp"].apply(lambda x: f"{x:+.2f}%")
display_df["Val_F1_Weighted"] = display_df["Val_F1_Weighted"].apply(lambda x: f"{x:.4f}")

display_df.columns = [
    "Configuration",
    "Threshold",
    "Zeroed Weights",
    "% Zeroed",
    "Val Accuracy",
    "Accuracy Change",
    "Val F1"
]

print("\n" + "=" * 95)
print("MACRO-SPARSITY ACCURACY BREAKDOWN & COLLAPSE TABLE")
print("=" * 95)
print(display_df.to_string(index=False))
print("=" * 95)

# --- Generate Academic Plot ---
plt.figure(figsize=(11, 6), dpi=300)
x_vals = results_df["Pct_Zeroed"]
y_vals = results_df["Val_Accuracy"]

plt.plot(x_vals, y_vals, marker="o", linewidth=2.5, color="#d62728", label="Model Validation Accuracy (%)")

# Highlight Selected Threshold (1e-3)
selected_row = results_df[results_df["Threshold"] == 0.001].iloc[0]
plt.scatter([selected_row["Pct_Zeroed"]], [selected_row["Val_Accuracy"]], color="#2ca02c", s=180, zorder=5, 
            label=f"Selected FYP Frontier τ=1e-3 (1.84% Zeroed, {selected_row['Val_Accuracy']:.2f}%)")

# Faculty Guideline Line
plt.axvline(x=2.0, color="#ff7f0e", linestyle="--", linewidth=1.8, label="Faculty Target Guideline (~2%)")

# Formatting
plt.title("Macro-Sparsity Analysis: Finding the Accuracy Collapse Point (1.84% to ~70%)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Model Zero-Sparsity (% of Total 1,091,188 Weights Zeroed)", fontsize=12)
plt.ylabel("Validation Accuracy (%)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower left", fontsize=10, frameon=True)

# Set dynamic Y limit based on collapse depth
min_y = min(y_vals)
plt.ylim([max(0.0, min_y - 5.0), 98.0])
plt.xlim([-2.0, max(x_vals) + 3.0])

plot_save_path = os.path.join(OUTPUT_DIR, "V5_MACRO_ACCURACY_COLLAPSE_CURVE.png")
plt.tight_layout()
plt.savefig(plot_save_path)
plt.close()
print(f"✓ Saved collapse plot to: {plot_save_path}")
print("=" * 95)
