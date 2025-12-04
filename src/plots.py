# plots.py
# ---------------------------------------------------------
# Generates plots:
# 1. Confusion Matrix
# 2. Feature Importance (RF or XGBoost)
# 3. PnL Curve (Using project-rule PnL)
# 4. Probability-based PnL Curve (if model gives predict_proba)
# ---------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


OUTPUT_DIR = "outputs"


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


# ---------------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    ensure_output_dir()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


# ---------------------------------------------------------
# 2. Feature Importance Plot
# Works for both RandomForest & XGBoost
# ---------------------------------------------------------
def plot_feature_importance(model, feature_names):
    ensure_output_dir()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        print("⚠ Model has no feature_importances_. Skipping plot.")
        return

    sorted_idx = np.argsort(importance)
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importance = importance[sorted_idx]

    plt.figure(figsize=(8, 10))
    plt.barh(sorted_features, sorted_importance)
    plt.title("Feature Importance")
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


# ---------------------------------------------------------
# 3. PnL Curve (project rule)
# ---------------------------------------------------------
def plot_pnl_curve(results):
    ensure_output_dir()

    plt.figure(figsize=(10, 5))
    plt.plot(results['model_pnl'], linewidth=1.5)
    plt.title("PnL Curve (Project Rule)")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)

    path = f"{OUTPUT_DIR}/pnl_curve.png"
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")


# ---------------------------------------------------------
# 4. Probability-based PnL Curve (optional)
# Example:
# If prob > 0.55 → buy
# Else → sell
# ---------------------------------------------------------
def plot_probability_pnl_curve(model, X_test_unscaled, X_test_scaled):
    ensure_output_dir()

    if not hasattr(model, "predict_proba"):
        print("⚠ Model does not support predict_proba. Skipping probability PnL curve.")
        return

    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Simple strategy: buy if prob > 0.55
    calls = np.where(probs > 0.55, "buy", "sell")
    prices = X_test_unscaled['Close'].values

    pnl = 0
    pnl_list = []

    for i in range(len(prices)):
        price = prices[i]
        if calls[i] == "buy":
            pnl -= price
        else:
            pnl += price
        pnl_list.append(pnl)

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_list, linewidth=1.5)
    plt.title("PnL Curve (Probability Strategy)")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)

    path = f"{OUTPUT_DIR}/pnl_curve_prob.png"
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")
