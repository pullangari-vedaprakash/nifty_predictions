# models.py
# ---------------------------------------------------------
# Train ML models and choose best performer
# ---------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_test_split_time(df, feature_cols, split_ratio):
    X = df[feature_cols]
    y = df['target']

    split = int(len(df) * split_ratio)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, X_test


def train_models(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    acc_rf = accuracy_score(y_test, pred_rf)
    acc_xgb = accuracy_score(y_test, pred_xgb)

    best_model = xgb if acc_xgb >= acc_rf else rf
    best_preds = pred_xgb if acc_xgb >= acc_rf else pred_rf
    model_name = "XGBoost" if acc_xgb >= acc_rf else "Random Forest"

    return best_model, best_preds, model_name
