"""
Script principal de ejecución por consola que coordina el pipeline completo:
Descarga (KaggleHub) -> Limpieza -> Transformación -> Entrenamiento -> Evaluación.
"""

import argparse
from pathlib import Path
from typing import Optional
from data_processing import (
    compute_rolling_features,
    download_kaggle_dataset,
    load_and_clean_data,
)
from evaluation import evaluate_multiclass_model, plot_confusion_matrix
from model import (
    chronological_train_test_split,
    save_artifacts,
    scale_features,
    train_logistic_regression,
    train_random_forest,
)


DEFAULT_ARTIFACTS_PATH = Path("artifacts/football_models.joblib")


def run_pipeline(
    data_path: Optional[Path] = None,
    division: Optional[str] = "E0",
    rolling_window: int = 5,
    test_size: float = 0.2,
    show_plots: bool = True,
) -> None:
    """
    Ejecuta el flujo de punta a punta del proyecto de predicción de fútbol.

    Args:
        data_path: Ruta opcional a un CSV local. Si es None, descarga desde KaggleHub.
        division: Código de liga (ej. 'E0' para Premier League, None para todas).
        rolling_window: Tamaño de la ventana móvil de partidos previos.
        test_size: Proporción del dataset destinada a evaluación cronológica.
        show_plots: Si es True, despliega las matrices de confusión en pantalla.
    """
    if data_path is None:
        print("1. Descargando / localizando dataset oficial desde KaggleHub...")
        data_path = download_kaggle_dataset()

    print(f"2. Cargando y limpiando datos desde '{data_path}' (División={division})...")
    raw_df = load_and_clean_data(filepath=data_path, division=division)

    print(f"3. Calculando variables históricas móviles (ventana={rolling_window})...")
    processed_df, feature_cols = compute_rolling_features(raw_df, window=rolling_window)
    print(f"   Partidos válidos tras Feature Engineering: {len(processed_df)}")
    print(f"   Features utilizadas ({len(feature_cols)}): {feature_cols}")

    print("4. Dividiendo en Train/Test respetando orden cronológico...")
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        processed_df, feature_cols=feature_cols, test_size=test_size
    )
    print(f"   Tamaño Train: {len(X_train)} | Tamaño Test: {len(X_test)}")

    print("5. Escalando características (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("6. Entrenando modelos (Logistic Regression Multinomial y Random Forest)...")
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)

    print("\n7. Evaluando modelos en conjunto de prueba...")
    logreg_metrics = evaluate_multiclass_model(
        logreg_model,
        X_test_scaled,
        y_test,
        model_name="Logistic Regression (Multinomial)",
    )
    plot_confusion_matrix(
        y_test,
        logreg_metrics["y_pred"],
        model_name="Logistic Regression (Multinomial)",
        save_path="artifacts/cm_logistic_regression.png",
        show_plot=show_plots,
    )

    rf_metrics = evaluate_multiclass_model(
        rf_model,
        X_test_scaled,
        y_test,
        model_name="Random Forest Classifier",
    )
    plot_confusion_matrix(
        y_test,
        rf_metrics["y_pred"],
        model_name="Random Forest Classifier",
        save_path="artifacts/cm_random_forest.png",
        show_plot=show_plots,
    )

    print("\n8. Guardando modelos y escalador entrenados...")
    save_artifacts(
        models={
            "Logistic Regression": logreg_model,
            "Random Forest": rf_model,
        },
        scaler=scaler,
        feature_cols=feature_cols,
        filepath=DEFAULT_ARTIFACTS_PATH,
    )
    print(f"   Pipeline finalizado con éxito. Artefactos en '{DEFAULT_ARTIFACTS_PATH}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento y evaluación para predicción de fútbol."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Ruta opcional a un archivo CSV local (por defecto descarga desde KaggleHub)",
    )
    parser.add_argument(
        "--division",
        type=str,
        default="E0",
        help="Liga a filtrar, ej. 'E0' (Premier League), 'SP1' (LaLiga) (default: E0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Cantidad de partidos para las medias móviles (default: 5)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventanas emergentes de matplotlib (solo guardar PNG)",
    )
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        division=args.division,
        rolling_window=args.window,
        show_plots=not args.no_show,
    )
