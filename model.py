"""
Módulo de división temporal, escalado, entrenamiento, predicción y serialización de modelos.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def chronological_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba respetando estrictamente
    el orden cronológico de los partidos (sin mezclar aleatoriamente).

    Args:
        df: DataFrame ordenado por fecha.
        feature_cols: Lista de columnas predictoras.
        target_col: Nombre de la columna objetivo.
        test_size: Proporción del final de la serie reservada para test (0 < test_size < 1).

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1.0 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Ajusta un StandardScaler únicamente sobre el conjunto de entrenamiento y
    transforma tanto entrenamiento como prueba.

    Args:
        X_train: Variables predictoras de entrenamiento.
        X_test: Variables predictoras de prueba.

    Returns:
        Tupla (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: pd.Series | np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Entrena un modelo de Regresión Logística Multinomial (Softmax) para clasificación
    multiclase (Local / Empate / Visitante).

    Args:
        X_train_scaled: Matriz de entrenamiento escalada.
        y_train: Vector de etiquetas de entrenamiento.
        max_iter: Iteraciones máximas del optimizador L-BFGS.
        random_state: Semilla para reproducibilidad.

    Returns:
        Modelo LogisticRegression entrenado.
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entrena un clasificador Random Forest para predecir el resultado del partido.

    Args:
        X_train: Matriz o DataFrame de entrenamiento.
        y_train: Vector de etiquetas de entrenamiento.
        n_estimators: Número de árboles en el ensamble.
        max_depth: Profundidad máxima de cada árbol para controlar sobreajuste.
        random_state: Semilla para reproducibilidad.

    Returns:
        Modelo RandomForestClassifier entrenado.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_match_probabilities(
    model: Any,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calcula las probabilidades de Victoria Local (H), Empate (D) y Victoria Visitante (A)
    para un conjunto de características de un partido.

    Args:
        model: Modelo entrenado con soporte para `predict_proba`.
        scaler: Instancia de StandardScaler ajustada en entrenamiento.
        features_df: DataFrame de una fila con las columnas predictoras en el orden esperado.

    Returns:
        Diccionario con probabilidades {"H": float, "D": float, "A": float}.
    """
    X_scaled = scaler.transform(features_df)
    probs = model.predict_proba(X_scaled)[0]
    return {
        "H": float(probs[0]),
        "D": float(probs[1]),
        "A": float(probs[2]),
    }


def save_artifacts(
    models: Dict[str, Any],
    scaler: StandardScaler,
    feature_cols: List[str],
    filepath: str | Path = "artifacts/football_models.joblib",
) -> None:
    """
    Serializa los modelos entrenados, el escalador y la lista de features en disco.

    Args:
        models: Diccionario de modelos entrenados (ej. {"logreg": ..., "rf": ...}).
        scaler: Escalador ajustado.
        feature_cols: Nombres de las columnas predictoras utilizadas.
        filepath: Ruta del archivo `.joblib` de salida.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": models,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    joblib.dump(payload, path)


def load_artifacts(
    filepath: str | Path = "artifacts/football_models.joblib",
) -> Dict[str, Any]:
    """
    Carga los artefactos serializados desde disco.

    Args:
        filepath: Ruta al archivo `.joblib`.

    Returns:
        Diccionario con claves `models`, `scaler` y `feature_cols`.
    """
    return joblib.load(filepath)

