# main.py
# ---------------------------------------------------------
# Full ML pipeline for NIFTY intraday direction prediction
# ---------------------------------------------------------

from config import DATA_PATH, FEATURE_COLS, TRAIN_SPLIT_RATIO
from data_loader import load_data
from features import add_features
from models import train_test_split_time, train_models
from evaluation import evaluate_model
from trading import compute_pnl
from plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pnl_curve,
    plot_probability_pnl_curve
)
import os


def main():

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("\n--- Loading Data ---")
    df = load_data(DATA_PATH)
    print(f"Rows Loaded: {len(df)}")

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("\n--- Feature Engineering ---")
    df = add_features(df)
    print(f"Rows After Features: {len(df)}")

    # ---------------------------------------------------------
    # 3. Train/Test Split
    # ---------------------------------------------------------
    print("\n--- Preparing Train/Test Split ---")
    X_train_s, X_test_s, y_train, y_test, X_test_unscaled = train_test_split_time(
        df, FEATURE_COLS, TRAIN_SPLIT_RATIO
    )

    print(f"Train Samples: {len(y_train)}")
    print(f"Test Samples:  {len(y_test)}")

    # -----------------------------------------------
    # Save Train & Test datasets
    # -----------------------------------------------
    print("\n--- Saving train_data.csv and test_data.csv ---")

    # ensure /data exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Train dataset = first part of DF
    train_df = df.iloc[:len(y_train)].copy()
    train_df.to_csv("data/train_data.csv", index=True)
    print("Saved: data/train_data.csv")

    # Test dataset = last part of DF
    test_df = df.iloc[len(y_train):].copy()
    test_df.to_csv("data/test_data.csv", index=True)
    print("Saved: data/test_data.csv")

    # ---------------------------------------------------------
    # 4. Train Models (RF & XGBoost)
    # ---------------------------------------------------------
    print("\n--- Training Models ---")
    model, preds, model_name = train_models(
        X_train_s, X_test_s, y_train, y_test
    )

    print(f"\nSelected Model: {model_name}")

    # ---------------------------------------------------------
    # 5. Evaluation Metrics
    # ---------------------------------------------------------
    print("\n--- Evaluation Metrics ---")
    evaluate_model(y_test, preds)

    # ---------------------------------------------------------
    # 6. Confusion Matrix Plot
    # ---------------------------------------------------------
    print("\n--- Saving Confusion Matrix ---")
    plot_confusion_matrix(y_test, preds)

    # ---------------------------------------------------------
    # 7. Feature Importance Plot
    # ---------------------------------------------------------
    print("\n--- Saving Feature Importance Plot ---")
    plot_feature_importance(model, FEATURE_COLS)

    # ---------------------------------------------------------
    # 8. Project-Rule PnL Calculation
    # ---------------------------------------------------------
    print("\n--- Computing Project-Rule PnL ---")
    results = compute_pnl(X_test_unscaled, preds)

    print("\nFirst 10 Results:")
    print(results.head(10))

    print("\nLast 10 Results:")
    print(results.tail(10))

    print(f"\nFinal Cumulative PnL: {results['model_pnl'].iloc[-1]:.2f}")

    # ---------------------------------------------------------
    # 9. PnL Curve Plot
    # ---------------------------------------------------------
    print("\n--- Saving PnL Curve ---")
    plot_pnl_curve(results)

    # ---------------------------------------------------------
    # 10. Probability-based PnL Curve
    # ---------------------------------------------------------
    print("\n--- Saving Probability-based PnL Curve ---")
    plot_probability_pnl_curve(model, X_test_unscaled, X_test_s)

    print("\nPipeline Completed Successfully! 🚀")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
