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

    print("\n--- Loading Data ---")
    df = load_data(DATA_PATH)
    print(f"Rows Loaded: {len(df)}")

    print("\n--- Feature Engineering ---")
    df = add_features(df)
    print(f"Rows After Features: {len(df)}")

    print("\n--- Preparing Train/Test Split ---")
    X_train_s, X_test_s, y_train, y_test, X_test_unscaled = train_test_split_time(
        df, FEATURE_COLS, TRAIN_SPLIT_RATIO
    )

    print(f"Train Samples: {len(y_train)}")
    print(f"Test Samples:  {len(y_test)}")

    print("\n--- Saving train_data.csv and test_data.csv ---")

    if not os.path.exists("data"):
        os.makedirs("data")

    train_df = df.iloc[:len(y_train)].copy()
    train_df_small = train_df.iloc[:50000].copy()
    train_df_small.to_csv("data/train_data.csv", index=True)
    print("Saved: data/train_data.csv (reduced rows)")

    test_df = df.iloc[len(y_train):].copy()
    test_df.to_csv("data/test_data.csv", index=True)
    print("Saved: data/test_data.csv")

    print("\n--- Training Models ---")
    model, preds, model_name = train_models(
        X_train_s, X_test_s, y_train, y_test
    )
    print(f"\nSelected Model: {model_name}")

    print("\n--- Evaluation Metrics ---")
    evaluate_model(y_test, preds)

    print("\n--- Saving Confusion Matrix ---")
    plot_confusion_matrix(y_test, preds)

    print("\n--- Saving Feature Importance Plot ---")
    plot_feature_importance(model, FEATURE_COLS)

    print("\n--- Computing Project-Rule PnL ---")
    results = compute_pnl(X_test_unscaled, preds)

    print("\nFirst 10 Results:")
    print(results.head(10))

    print("\nLast 10 Results:")
    print(results.tail(10))

    print(f"\nFinal Cumulative PnL: {results['model_pnl'].iloc[-1]:.2f}")

    print("\n--- Saving PnL Curve ---")
    plot_pnl_curve(results)

    print("\n--- Saving Probability-based PnL Curve ---")
    plot_probability_pnl_curve(model, X_test_unscaled, X_test_s)

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()
