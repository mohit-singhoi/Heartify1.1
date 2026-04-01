# =============================================================================
#  Heart Attack Risk Prediction — Model Testing & Diagnostics
#  model_test.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, average_precision_score
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

RANDOM_STATE = 42
DATA_PATH    = "Datasets/heart_attack_risk_dataset_20k.csv"
MODEL_DIR    = "models"
REPORT_DIR   = "reports"

print("=" * 70)
print("  HEART ATTACK RISK PREDICTION — MODEL TESTING")
print("=" * 70)

# ── 1. Load artefacts ─────────────────────────────────────────────────────────
print("\n[1/5] Loading saved models & scaler...")

scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM":                 "svm.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
}
models = {}
needs_scaling = {"Logistic Regression": True, "SVM": True,
                 "Decision Tree": False, "Random Forest": False}

for name, fname in model_files.items():
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"  ✅ {name} loaded")
    else:
        print(f"  ❌ {name} NOT FOUND — run train_models.py first")

if not models:
    raise FileNotFoundError("No models found. Run train_models.py first.")

# ── 2. Prepare test data ──────────────────────────────────────────────────────
print("\n[2/5] Preparing test data...")

df = pd.read_csv(DATA_PATH)
df.drop(columns=["patient_id", "risk_probability", "risk_category"], inplace=True)

le = LabelEncoder()
for col in ["gender", "physical_activity", "diet_quality"]:
    df[col] = le.fit_transform(df[col])

TARGET = "heart_attack_risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_test_sc = scaler.transform(X_test)
print(f"  Test samples : {X_test.shape[0]}")
print(f"  Positive     : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── 3. Evaluate all models ────────────────────────────────────────────────────
print("\n[3/5] Running predictions on test set...")

all_results   = {}
all_probas    = {}
COLORS = {
    "Logistic Regression": "#4C72B0",
    "SVM":                 "#DD8452",
    "Decision Tree":       "#55A868",
    "Random Forest":       "#C44E52",
}

for name, mdl in models.items():
    Xte    = X_test_sc if needs_scaling[name] else X_test
    y_pred = mdl.predict(Xte)
    y_prob = mdl.predict_proba(Xte)[:, 1]
    all_probas[name] = y_prob

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)   # specificity
    sens = tp / (tp + fn)   # sensitivity / recall

    all_results[name] = {
        "Accuracy":    acc,
        "F1 Score":    f1,
        "ROC-AUC":     auc,
        "Avg Prec":    ap,
        "Sensitivity": sens,
        "Specificity": spec,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

    print(f"\n  ── {name} ──")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    F1 Score     : {f1:.4f}")
    print(f"    ROC-AUC      : {auc:.4f}")
    print(f"    Sensitivity  : {sens:.4f}  (True Positive Rate)")
    print(f"    Specificity  : {spec:.4f}  (True Negative Rate)")
    print(f"    TP={tp} | TN={tn} | FP={fp} | FN={fn}")

# ── 4. Winner announcement ────────────────────────────────────────────────────
print("\n[4/5] Determining best model...")

df_res = pd.DataFrame(all_results).T.sort_values("ROC-AUC", ascending=False)
best_name = df_res["ROC-AUC"].idxmax()

print("\n  ╔══ TEST SET SCORES (sorted by ROC-AUC) ══╗")
display_cols = ["Accuracy","F1 Score","ROC-AUC","Sensitivity","Specificity"]
print(df_res[display_cols].round(4).to_string())
print(f"\n  🏆 Best Model: {best_name}  (ROC-AUC = {df_res.loc[best_name,'ROC-AUC']:.4f})")

# ── 5. Test plots ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating test diagnostic plots...")

os.makedirs(REPORT_DIR, exist_ok=True)

# ── 5a. Probability Distribution per model ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Predicted Probability Distributions (Test Set)", fontsize=14, fontweight="bold")

for ax, name in zip(axes.flat, models.keys()):
    prob = all_probas[name]
    ax.hist(prob[y_test == 0], bins=50, alpha=0.6, color="steelblue", label="No Risk")
    ax.hist(prob[y_test == 1], bins=50, alpha=0.6, color="tomato",    label="Heart Attack")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "prob_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 5b. Sensitivity vs Specificity Radar-style bar ──────────────────────────
names_sorted = list(df_res.index)
x  = np.arange(len(names_sorted))
w  = 0.2
metrics_bar = ["Accuracy","F1 Score","ROC-AUC","Sensitivity","Specificity"]
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(13, 6))
for i, met in enumerate(metrics_bar):
    vals = [all_results[n][met] for n in names_sorted]
    ax.bar(x + i*w, vals, width=w, label=met, color=cmap(i), edgecolor="white")

ax.set_xticks(x + w*2)
ax.set_xticklabels(names_sorted, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Metrics — All Models (Test Set)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "test_metrics_grouped.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 5c. ROC overlay ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],"k--", lw=1.2)
for name in names_sorted:
    fpr, tpr, _ = roc_curve(y_test, all_probas[name])
    auc = all_results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, lw=2.5, color=COLORS[name], label=f"{name} (AUC={auc:.4f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "test_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 5d. Single-sample walkthrough ────────────────────────────────────────────
print("\n  ── Single Sample Prediction Test ──")
sample = X_test.iloc[[0]]
actual = y_test.iloc[0]
print(f"\n  Actual label: {'❤️  HEART ATTACK RISK' if actual==1 else '✅  NO RISK'}")

for name, mdl in models.items():
    Xs = scaler.transform(sample) if needs_scaling[name] else sample
    pred = mdl.predict(Xs)[0]
    prob = mdl.predict_proba(Xs)[0][1]
    status = "✅ CORRECT" if pred == actual else "❌ WRONG"
    print(f"  {name:22s} → Pred: {'RISK' if pred==1 else 'NO RISK':8s}  "
          f"Prob: {prob:.4f}  {status}")

print("\n  ✅ All test plots saved to reports/")
print("\n" + "=" * 70)
print("  TEST COMPLETE")
print(f"  🏆 Recommended Model: {best_name}")
print("=" * 70)
print("\n  Next step: Run  streamlit run app.py")
