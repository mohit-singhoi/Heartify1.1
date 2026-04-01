# =============================================================================
#  Heart Attack Risk Prediction — Model Training Pipeline
#  train_models.py
#  Models: Logistic Regression | SVM | Decision Tree | Random Forest
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import time

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ── 0. Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = "Datasets/heart_attack_risk_dataset_20k.csv"
OUTPUT_DIR  = "models"
REPORT_DIR  = "reports"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("  HEART ATTACK RISK PREDICTION — MODEL TRAINING PIPELINE")
print("=" * 70)

# ── 1. Load & Inspect ─────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape          : {df.shape}")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Target dist.   :\n{df['heart_attack_risk'].value_counts().to_string()}")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("\n[2/7] Preprocessing...")

# Drop non-predictive columns
df.drop(columns=["patient_id", "risk_probability", "risk_category"], inplace=True)

# Encode categoricals
le = LabelEncoder()
categorical_cols = ["gender", "physical_activity", "diet_quality"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

TARGET = "heart_attack_risk"
X = df.drop(columns=[TARGET])
y = df[TARGET]
FEATURE_NAMES = X.columns.tolist()
print(f"  Features  : {X.shape[1]}")
print(f"  Samples   : {X.shape[0]}")

# Train/Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features (fit on train only)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Save scaler & feature names
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(FEATURE_NAMES, os.path.join(OUTPUT_DIR, "feature_names.pkl"))
print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 3. Define Models ──────────────────────────────────────────────────────────
print("\n[3/7] Defining models...")

models = {
    "Logistic Regression": {
        "model": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "scaled": True,
        "color": "#4C72B0"
    },
    "SVM": {
        "model": SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "scaled": True,
        "color": "#DD8452"
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            criterion="gini", random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "scaled": False,
        "color": "#55A868"
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced"
        ),
        "scaled": False,
        "color": "#C44E52"
    },
}

# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
print("\n[4/7] Training & evaluating models...")

results     = {}
trained     = {}
cv_scores   = {}
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, cfg in models.items():
    print(f"\n  ── {name} ──")
    mdl = cfg["model"]
    Xtr = X_train_sc if cfg["scaled"] else X_train
    Xte = X_test_sc  if cfg["scaled"] else X_test

    # Train
    t0 = time.time()
    mdl.fit(Xtr, y_train)
    elapsed = time.time() - t0

    # Predict
    y_pred  = mdl.predict(Xte)
    y_proba = mdl.predict_proba(Xte)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    ap   = average_precision_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    # Cross-validation (on train set)
    X_cv = X_train_sc if cfg["scaled"] else X_train
    cv_f1 = cross_val_score(mdl, X_cv, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_scores[name] = cv_f1

    results[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1 Score":  f1,
        "ROC-AUC":   auc,
        "Avg Precision": ap,
        "CV F1 Mean": cv_f1.mean(),
        "CV F1 Std":  cv_f1.std(),
        "Train Time (s)": round(elapsed, 2),
    }
    trained[name] = {"model": mdl, "scaled": cfg["scaled"], "color": cfg["color"],
                     "y_pred": y_pred, "y_proba": y_proba, "cm": cm}

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    CV F1     : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"    Train Time: {elapsed:.2f}s")

    # Save model
    joblib.dump(mdl, os.path.join(OUTPUT_DIR, f"{name.replace(' ','_').lower()}.pkl"))
    print(f"    Saved → models/{name.replace(' ','_').lower()}.pkl")

# ── 5. Comparison Table ───────────────────────────────────────────────────────
print("\n[5/7] Building comparison report...")

results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
results_df = results_df.round(4)
results_df.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"))

print("\n  ╔══ MODEL COMPARISON (sorted by ROC-AUC) ══╗")
print(results_df[["Accuracy","Precision","Recall","F1 Score","ROC-AUC","CV F1 Mean"]].to_string())

best_model_name = results_df["ROC-AUC"].idxmax()
print(f"\n  🏆 Best Model: {best_model_name}  (ROC-AUC = {results_df.loc[best_model_name,'ROC-AUC']:.4f})")

# ── 6. Visualisations ─────────────────────────────────────────────────────────
print("\n[6/7] Generating visualisations...")

colors = [trained[n]["color"] for n in results_df.index]
names  = list(results_df.index)

# --- 6a. Metrics Bar Chart ---------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Heart Attack Risk Prediction — Model Comparison", fontsize=16, fontweight="bold")
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV F1 Mean"]

for ax, metric in zip(axes.flat, metrics):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6b. Confusion Matrices ---------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
for ax, name in zip(axes, names):
    cm = trained[name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Risk","Risk"], yticklabels=["No Risk","Risk"],
                linewidths=0.5, cbar=False)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6c. ROC Curves ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Random (AUC = 0.50)")
for name in names:
    fpr, tpr, _ = roc_curve(y_test, trained[name]["y_proba"])
    auc = results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, lw=2, color=trained[name]["color"], label=f"{name} (AUC = {auc:.4f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6d. Precision-Recall Curves ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
for name in names:
    prec_c, rec_c, _ = precision_recall_curve(y_test, trained[name]["y_proba"])
    ap = results[name]["Avg Precision"]
    ax.plot(rec_c, prec_c, lw=2, color=trained[name]["color"], label=f"{name} (AP = {ap:.4f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6e. CV Score Boxplot ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
cv_data = [cv_scores[n] for n in names]
bp = ax.boxplot(cv_data, patch_artist=True, notch=False, vert=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(names, rotation=10, fontsize=11)
ax.set_ylabel("F1 Score (CV)", fontsize=12)
ax.set_title("5-Fold Cross-Validation F1 Scores", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "cv_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6f. Feature Importance (Random Forest + Permutation for LR) -------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Random Forest native importance
rf = trained["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
top15 = importances.tail(15)
top15.plot(kind="barh", ax=axes[0], color="#C44E52", edgecolor="white")
axes[0].set_title("Random Forest — Feature Importance (Top 15)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Importance Score")

# Logistic Regression coefficients
lr = trained["Logistic Regression"]["model"]
coef = pd.Series(np.abs(lr.coef_[0]), index=FEATURE_NAMES).sort_values(ascending=True)
top15_lr = coef.tail(15)
top15_lr.plot(kind="barh", ax=axes[1], color="#4C72B0", edgecolor="white")
axes[1].set_title("Logistic Regression — |Coefficient| (Top 15)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

print("  ✅ All plots saved to reports/")

# ── 7. Classification Reports ─────────────────────────────────────────────────
print("\n[7/7] Saving classification reports...")

with open(os.path.join(REPORT_DIR, "classification_reports.txt"), "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  HEART ATTACK RISK — DETAILED CLASSIFICATION REPORTS\n")
    f.write("=" * 70 + "\n\n")
    for name in names:
        f.write(f"{'─'*50}\n")
        f.write(f"  Model: {name}\n")
        f.write(f"{'─'*50}\n")
        Xte = X_test_sc if trained[name]["scaled"] else X_test
        report = classification_report(y_test, trained[name]["y_pred"],
                                       target_names=["No Risk", "Heart Attack Risk"])
        f.write(report + "\n")

print("  ✅ reports/classification_reports.txt")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"\n  🏆 Best Model : {best_model_name}")
print(f"     ROC-AUC   : {results[best_model_name]['ROC-AUC']:.4f}")
print(f"     F1 Score  : {results[best_model_name]['F1 Score']:.4f}")
print(f"     Accuracy  : {results[best_model_name]['Accuracy']:.4f}")
print(f"\n  📁 Saved artifacts:")
print(f"     models/  → 4 model .pkl files + scaler.pkl + feature_names.pkl")
print(f"     reports/ → CSV comparison, 6 plots, classification reports")
print("\n  Next step: Run  python model_test.py")
print("=" * 70)
