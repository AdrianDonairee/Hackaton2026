"""
Interfaz web interactiva con Streamlit para predecir probabilidades de partidos
de fútbol (Victoria Local H, Empate D, Victoria Visitante A).
"""

from typing import Any, Dict, Tuple
import pandas as pd
import streamlit as st
from data_processing import (
    compute_rolling_features,
    download_kaggle_dataset,
    load_and_clean_data,
)
from main import DEFAULT_ARTIFACTS_PATH, run_pipeline
from model import load_artifacts, predict_match_probabilities


st.set_page_config(
    page_title="Predicción de Fútbol - ML",
    page_icon="⚽",
    layout="centered",
)


@st.cache_resource
def get_trained_artifacts_and_data() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Descarga/localiza el dataset de KaggleHub y carga los modelos serializados.
    Si los modelos aún no existen en disco, ejecuta el pipeline inicial.
    """
    csv_path = download_kaggle_dataset()

    if not DEFAULT_ARTIFACTS_PATH.exists():
        run_pipeline(data_path=csv_path, show_plots=False)

    artifacts = load_artifacts(DEFAULT_ARTIFACTS_PATH)
    raw_df = load_and_clean_data(filepath=csv_path)
    processed_df, _ = compute_rolling_features(raw_df, window=5)
    return artifacts, processed_df


def get_latest_team_stats(df: pd.DataFrame, team: str) -> Dict[str, float]:
    """
    Obtiene las últimas estadísticas móviles registradas de un equipo en el dataset.
    """
    team_rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if team_rows.empty:
        return {"GF": 1.4, "GA": 1.2, "Pts": 1.5, "SOT": 4.0}

    last_row = team_rows.iloc[-1]
    if last_row["HomeTeam"] == team:
        return {
            "GF": float(last_row["Home_Roll_GF"]),
            "GA": float(last_row["Home_Roll_GA"]),
            "Pts": float(last_row["Home_Roll_Pts"]),
            "SOT": float(last_row.get("Home_Roll_SOT", 4.0)),
        }
    return {
        "GF": float(last_row["Away_Roll_GF"]),
        "GA": float(last_row["Away_Roll_GA"]),
        "Pts": float(last_row["Away_Roll_Pts"]),
        "SOT": float(last_row.get("Away_Roll_SOT", 4.0)),
    }


def main() -> None:
    """Renderiza la aplicación interactiva en Streamlit."""
    st.title("⚽ Predicción de Resultados de Fútbol")
    st.markdown(
        "Estimación de probabilidades **Local (H)**, **Empate (D)** y **Visitante (A)** "
        "basada en el dataset `adamgbor/club-football-match-data-2000-2025` de Kaggle."
    )

    artifacts, processed_df = get_trained_artifacts_and_data()
    models = artifacts["models"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    teams = sorted(processed_df["HomeTeam"].unique().tolist())

    st.sidebar.header("Configuración del Modelo")
    selected_model_name = st.sidebar.selectbox(
        "Seleccionar Clasificador",
        options=list(models.keys()),
    )
    selected_model = models[selected_model_name]

    input_mode = st.sidebar.radio(
        "Modo de ingreso de datos",
        options=["Seleccionar Equipos", "Ingresar Parámetros Manualmente"],
    )

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("Equipo Local", options=teams, index=0)
    with col_away:
        default_away_idx = 1 if len(teams) > 1 else 0
        away_team = st.selectbox("Equipo Visitante", options=teams, index=default_away_idx)

    if home_team == away_team:
        st.warning("Seleccioná dos equipos distintos para realizar la predicción.")
        return

    home_defaults = get_latest_team_stats(processed_df, home_team)
    away_defaults = get_latest_team_stats(processed_df, away_team)

    disabled = input_mode == "Seleccionar Equipos"

    st.subheader("Promedios Móviles (Últimos 5 partidos)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{home_team} (Local)**")
        home_gf = st.number_input(
            "Goles a Favor (Local)", 0.0, 5.0, home_defaults["GF"], 0.1, disabled=disabled
        )
        home_ga = st.number_input(
            "Goles en Contra (Local)", 0.0, 5.0, home_defaults["GA"], 0.1, disabled=disabled
        )
        home_pts = st.number_input(
            "Puntos por Partido (Local)", 0.0, 3.0, home_defaults["Pts"], 0.1, disabled=disabled
        )
        home_sot = st.number_input(
            "Tiros al Arco (Local)", 0.0, 15.0, home_defaults["SOT"], 0.5, disabled=disabled
        )

    with c2:
        st.markdown(f"**{away_team} (Visitante)**")
        away_gf = st.number_input(
            "Goles a Favor (Visitante)", 0.0, 5.0, away_defaults["GF"], 0.1, disabled=disabled
        )
        away_ga = st.number_input(
            "Goles en Contra (Visitante)", 0.0, 5.0, away_defaults["GA"], 0.1, disabled=disabled
        )
        away_pts = st.number_input(
            "Puntos por Partido (Visitante)", 0.0, 3.0, away_defaults["Pts"], 0.1, disabled=disabled
        )
        away_sot = st.number_input(
            "Tiros al Arco (Visitante)", 0.0, 15.0, away_defaults["SOT"], 0.5, disabled=disabled
        )

    feature_values = {
        "Home_Roll_GF": home_gf,
        "Home_Roll_GA": home_ga,
        "Home_Roll_Pts": home_pts,
        "Away_Roll_GF": away_gf,
        "Away_Roll_GA": away_ga,
        "Away_Roll_Pts": away_pts,
        "Diff_Roll_Pts": home_pts - away_pts,
        "Diff_Roll_GF": home_gf - away_gf,
        "Home_Roll_SOT": home_sot,
        "Away_Roll_SOT": away_sot,
    }

    if all(col in feature_cols for col in ["B365H", "B365D", "B365A"]):
        st.subheader("Cuotas de Referencia (Odds)")
        o1, o2, o3 = st.columns(3)
        feature_values["B365H"] = o1.number_input("Cuota Local (OddHome)", 1.01, 20.0, 2.10, 0.05)
        feature_values["B365D"] = o2.number_input("Cuota Empate (OddDraw)", 1.01, 20.0, 3.40, 0.05)
        feature_values["B365A"] = o3.number_input("Cuota Visitante (OddAway)", 1.01, 20.0, 3.20, 0.05)

    if st.button("Predecir Resultado", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{col: feature_values[col] for col in feature_cols}])
        probs = predict_match_probabilities(selected_model, scaler, input_df)

        st.divider()
        st.subheader(f"Probabilidades Estimadas ({selected_model_name})")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Victoria {home_team} (H)", f"{probs['H'] * 100:.1f}%")
        m2.metric("Empate (D)", f"{probs['D'] * 100:.1f}%")
        m3.metric(f"Victoria {away_team} (A)", f"{probs['A'] * 100:.1f}%")

        chart_df = pd.DataFrame(
            {
                "Resultado": [f"Local ({home_team})", "Empate", f"Visitante ({away_team})"],
                "Probabilidad": [probs["H"], probs["D"], probs["A"]],
            }
        ).set_index("Resultado")
        st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
