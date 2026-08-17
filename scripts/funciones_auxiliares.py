import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def is_integer_like(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    if not np.issubdtype(s.dtype, np.number):
        return False
    return np.allclose(s, np.round(s))


def main() -> None:
    seed = 13

    df = pd.read_csv("BBDD_ML_TAREA.csv")

    rename_dict = {
        "V1": "state",
        "V2": "account_length",
        "V3": "area_code",
        "V4": "phone_number",
        "V5": "international_plan",
        "V6": "voice_mail_plan",
        "V7": "number_vmail_messages",
        "V8": "total_day_minutes",
        "V9": "total_day_calls",
        "V10": "total_day_charge",
        "V11": "total_evening_minutes",
        "V12": "total_evening_calls",
        "V13": "total_evening_charge",
        "V14": "total_night_minutes",
        "V15": "total_night_calls",
        "V16": "total_night_charge",
        "V17": "total_international_minutes",
        "V18": "total_international_calls",
        "V19": "total_international_charge",
        "V20": "customer_service_calls",
        "Y": "y",
    }
    df = df.rename(columns=rename_dict)
    df = df.drop_duplicates().reset_index(drop=True)

    if "phone_number" in df.columns:
        df = df.drop(columns=["phone_number"])

    target = "y"
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Variables tipo codigo (nominales, sin orden): se codifican con OHE.
    code_like_cols = [c for c in ["state", "area_code"] if c in X.columns]

    # Variables binarias 0/1: ni escalado ni OHE.
    binary_cols = []
    for c in X.columns:
        s = X[c]
        if np.issubdtype(s.dropna().dtype, np.number):
            vals = set(pd.Series(s.dropna()).astype(float).unique().tolist())
            if vals.issubset({0.0, 1.0}):
                binary_cols.append(c)

    # Variables discretas enteras con sentido ordinal (cuentas): no OHE, no escalado.
    # Se excluyen las variables de codigo y las binarias.
    ordinal_int_cols = []
    for c in X.columns:
        if c in code_like_cols or c in binary_cols:
            continue
        if is_integer_like(X[c]):
            ordinal_int_cols.append(c)

    # Variables continuas: escalado (solo si no son binarias).
    continuous_cols = [
        c
        for c in X.columns
        if c not in code_like_cols and c not in binary_cols and c not in ordinal_int_cols
    ]

    # Tratamiento de perdidos SIN imputacion estadistica.
    # 1) Añadimos indicadores de perdido para todas las columnas.
    for c in X.columns:
        X[f"{c}__missing"] = X[c].isna().astype(int)

    # 2) Categorias de perdido.
    #    - En variables nominales (OHE): etiqueta explicita "MISSING".
    #    - En variables numericas: valor centinela (-1 o -999) + indicador de perdido.
    for c in code_like_cols:
        X[c] = X[c].astype("string").fillna("MISSING")

    for c in binary_cols + ordinal_int_cols:
        X[c] = X[c].fillna(-1)

    for c in continuous_cols:
        X[c] = X[c].fillna(-999.0)

    missing_flag_cols = [c for c in X.columns if c.endswith("__missing")]

    print("\nColumnas usadas por bloque:")
    print("- code_like_cols (OHE):", code_like_cols)
    print("- binary_cols (passthrough):", binary_cols)
    print("- ordinal_int_cols (passthrough):", ordinal_int_cols)
    print("- continuous_cols (StandardScaler):", continuous_cols)
    print("- missing_flag_cols (passthrough):", len(missing_flag_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), continuous_cols),
            (
                "ohe_codes",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                code_like_cols,
            ),
            ("binary", "passthrough", binary_cols),
            ("ordinal_int", "passthrough", ordinal_int_cols),
            ("missing_flags", "passthrough", missing_flag_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("col_transf", preprocessor),
            (
                "var_selection",
                RFE(
                    estimator=LogisticRegression(random_state=seed, max_iter=1000),
                    n_features_to_select=5,
                ),
            ),
            (
                "Red_neuronal",
                MLPClassifier(random_state=seed, early_stopping=True, max_iter=1000),
            ),
        ]
    )

    param_grid = {
        "Red_neuronal__hidden_layer_sizes": [(2,), (3,), (4,), (5,)],
        "Red_neuronal__activation": ["relu", "tanh"],
        "Red_neuronal__alpha": [1e-4, 1e-3, 1e-2],
        "Red_neuronal__learning_rate_init": [1e-3, 1e-2],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=10,
        error_score="raise",
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    grid.fit(X_train, y_train)

    print("\nBest params:", grid.best_params_)
    print("Best CV score:", round(grid.best_score_, 6))

    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", round(test_acc, 6))
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=4))

    # Export opcional
    # import joblib
    # joblib.dump(grid, "grid_v3_mlp.pkl")


if __name__ == "__main__":
    main()
