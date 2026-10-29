"""
Script principal de ejecución por consola que coordina el pipeline completo:
Descarga (KaggleHub) -> Limpieza -> Transformación -> Entrenamiento -> Evaluación -> Predicción Interactiva.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from data_processing import (
    LEAGUE_NAMES,
    compute_rolling_features,
    download_kaggle_dataset,
    get_latest_team_stats,
    load_and_clean_data,
)
from evaluation import evaluate_multiclass_model, plot_confusion_matrix
from model import (
    chronological_train_test_split,
    predict_match_probabilities,
    save_artifacts,
    scale_features,
    train_logistic_regression,
    train_random_forest,
)


DEFAULT_ARTIFACTS_PATH = Path("artifacts/football_models.joblib")


def select_team_in_console(teams: List[str], role_label: str) -> Optional[str]:
    """
    Permite buscar y elegir un equipo por consola escribiendo su nombre o parte de él.
    Devuelve None si el usuario presiona Enter vacío para salir.
    """
    while True:
        query = input(f"\nIngresá el nombre del {role_label} (o Enter para salir): ").strip()
        if not query:
            return None

        # Coincidencia exacta (ignorando mayúsculas/minúsculas)
        exact = [t for t in teams if t.lower() == query.lower()]
        if len(exact) == 1:
            return exact[0]

        # Coincidencias parciales
        matches = [t for t in teams if query.lower() in t.lower()]
        if not matches:
            print(f"   No se encontraron equipos que coincidan con '{query}'. Intentá de nuevo.")
            continue

        if len(matches) == 1:
            print(f"   Seleccionado: {matches[0]}")
            return matches[0]

        print("   Equipos encontrados:")
        for idx, team_name in enumerate(matches[:15], start=1):
            print(f"     {idx}. {team_name}")

        choice = input("   Elegí el número de equipo (o Enter para buscar de nuevo): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matches[:15]):
            return matches[int(choice) - 1]


def interactive_console_prediction(
    processed_df: pd.DataFrame,
    models: Dict[str, Any],
    scaler: Any,
    feature_cols: List[str],
) -> None:
    """
    Inicia un bucle interactivo en consola para seleccionar equipos y predecir
    las probabilidades de Victoria Local (H), Empate (D) y Victoria Visitante (A).
    """
    all_teams = sorted(processed_df["HomeTeam"].unique().tolist())

    print("\n" + "=" * 65)
    print("PREDICCIÓN INTERACTIVA DE PARTIDOS POR CONSOLA")
    print("=" * 65)
    print(f"Equipos disponibles en todas las ligas: {len(all_teams)}")
    print("Ejemplos: Real Madrid, Barcelona, Ath Madrid, Arsenal, Man City, Inter, Juventus...")

    while True:
        home_team = select_team_in_console(all_teams, "Equipo Local")
        if home_team is None:
            print("Saliendo del predictor por consola.")
            break

        away_team = select_team_in_console(all_teams, "Equipo Visitante")
        if away_team is None:
            print("Saliendo del predictor por consola.")
            break

        if home_team == away_team:
            print("   [!] Elegí dos equipos distintos.")
            continue

        h_stats = get_latest_team_stats(processed_df, home_team)
        a_stats = get_latest_team_stats(processed_df, away_team)

        feature_values = {
            "Home_Roll_GF": h_stats["GF"],
            "Home_Roll_GA": h_stats["GA"],
            "Home_Roll_Pts": h_stats["Pts"],
            "Away_Roll_GF": a_stats["GF"],
            "Away_Roll_GA": a_stats["GA"],
            "Away_Roll_Pts": a_stats["Pts"],
            "Diff_Roll_Pts": h_stats["Pts"] - a_stats["Pts"],
            "Diff_Roll_GF": h_stats["GF"] - a_stats["GF"],
            "Home_Roll_SOT": h_stats["SOT"],
            "Away_Roll_SOT": a_stats["SOT"],
            "B365H": 2.10,
            "B365D": 3.40,
            "B365A": 3.20,
        }

        input_df = pd.DataFrame([{col: feature_values[col] for col in feature_cols}])

        print("\n" + "-" * 65)
        print(f"PARTIDO: {home_team} (Local) vs {away_team} (Visitante)")
        print(
            f"Promedios (últ. 5): "
            f"{home_team} [GF={h_stats['GF']}, GA={h_stats['GA']}, Pts={h_stats['Pts']}] | "
            f"{away_team} [GF={a_stats['GF']}, GA={a_stats['GA']}, Pts={a_stats['Pts']}]"
        )
        print("-" * 65)

        for model_name, clf in models.items():
            probs = predict_match_probabilities(clf, scaler, input_df)
            print(
                f"[{model_name:20s}] -> "
                f"Local (H): {probs['H']*100:5.1f}% | "
                f"Empate (D): {probs['D']*100:5.1f}% | "
                f"Visitante (A): {probs['A']*100:5.1f}%"
            )
        print("-" * 65)


def run_pipeline(
    data_path: Optional[Path] = None,
    division: Optional[str] = None,
    rolling_window: int = 5,
    test_size: float = 0.2,
    show_plots: bool = True,
    interactive: bool = True,
) -> None:
    """
    Ejecuta el flujo de punta a punta del proyecto de predicción de fútbol.

    Args:
        data_path: Ruta opcional a un CSV local. Si es None, descarga desde KaggleHub.
        division: Código de liga opcional (ej. 'SP1' LaLiga, 'E0' Premier, o None para todas).
        rolling_window: Tamaño de la ventana móvil de partidos previos.
        test_size: Proporción del dataset destinada a evaluación cronológica.
        show_plots: Si es True, despliega las matrices de confusión en pantalla.
        interactive: Si es True, abre el predictor interactivo en consola al finalizar.
    """
    if data_path is None:
        print("1. Descargando / localizando dataset oficial desde KaggleHub...")
        data_path = download_kaggle_dataset()

    liga_desc = LEAGUE_NAMES.get(division, division) if division else "Todas las ligas"
    print(f"2. Cargando y limpiando datos desde '{data_path}' ({liga_desc})...")
    raw_df = load_and_clean_data(filepath=data_path, division=division)

    print(f"3. Calculando variables históricas móviles (ventana={rolling_window})...")
    processed_df, feature_cols = compute_rolling_features(raw_df, window=rolling_window)
    print(f"   Partidos válidos tras Feature Engineering: {len(processed_df)}")
    print(f"   Equipos únicos cargados: {processed_df['HomeTeam'].nunique()}")
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

    models_dict = {
        "Logistic Regression": logreg_model,
        "Random Forest": rf_model,
    }

    print("\n8. Guardando modelos y escalador entrenados...")
    save_artifacts(
        models=models_dict,
        scaler=scaler,
        feature_cols=feature_cols,
        filepath=DEFAULT_ARTIFACTS_PATH,
    )
    print(f"   Pipeline finalizado con éxito. Artefactos en '{DEFAULT_ARTIFACTS_PATH}'.")

    if interactive:
        interactive_console_prediction(
            processed_df=processed_df,
            models=models_dict,
            scaler=scaler,
            feature_cols=feature_cols,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento, evaluación y predicción de fútbol."
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
        default=None,
        help="Liga opcional a filtrar, ej. 'SP1' (LaLiga), 'E0' (Premier) (default: Todas)",
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
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Finalizar el script sin abrir el menú interactivo de consola",
    )
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        division=args.division,
        rolling_window=args.window,
        show_plots=not args.no_show,
        interactive=not args.no_interactive,
    )
