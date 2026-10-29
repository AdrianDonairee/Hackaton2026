"""
Módulo de descarga (KaggleHub), carga, limpieza e ingeniería de características
para el dataset de fútbol (Football-Data.co.uk / Club Football Match Data 2000-2025).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import kagglehub
import numpy as np
import pandas as pd


KAGGLE_DATASET_HANDLE = "adamgbor/club-football-match-data-2000-2025"

# Mapeo de columnas del dataset de Kaggle (adamgbor) al estándar Football-Data.co.uk
COLUMN_RENAME_MAP = {
    "MatchDate": "Date",
    "FTHome": "FTHG",
    "FTAway": "FTAG",
    "FTResult": "FTR",
    "HomeShots": "HS",
    "AwayShots": "AS",
    "HomeTarget": "HST",
    "AwayTarget": "AST",
    "HomeCorners": "HC",
    "AwayCorners": "AC",
    "OddHome": "B365H",
    "OddDraw": "B365D",
    "OddAway": "B365A",
}

# Nombres descriptivos de las principales ligas del dataset
LEAGUE_NAMES: Dict[str, str] = {
    "SP1": "LaLiga (España)",
    "E0": "Premier League (Inglaterra)",
    "I1": "Serie A (Italia)",
    "D1": "Bundesliga (Alemania)",
    "F1": "Ligue 1 (Francia)",
    "ARG": "Primera División (Argentina)",
    "BRA": "Brasileirão (Brasil)",
    "P1": "Primeira Liga (Portugal)",
    "N1": "Eredivisie (Países Bajos)",
    "B1": "Pro League (Bélgica)",
    "T1": "Süper Lig (Turquía)",
    "SC0": "Premiership (Escocia)",
    "E1": "Championship (Inglaterra 2ª)",
    "SP2": "LaLiga Hypermotion (España 2ª)",
    "I2": "Serie B (Italia 2ª)",
    "D2": "2. Bundesliga (Alemania 2ª)",
    "F2": "Ligue 2 (Francia 2ª)",
}

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
OPTIONAL_STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC"]
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
INVERSE_TARGET_MAP = {0: "H", 1: "D", 2: "A"}


def download_kaggle_dataset(
    dataset_handle: str = KAGGLE_DATASET_HANDLE,
) -> Path:
    """
    Descarga la última versión del dataset desde Kaggle usando `kagglehub`
    y devuelve la ruta al archivo CSV principal de partidos (`Matches.csv`).

    Args:
        dataset_handle: Identificador del dataset en Kaggle.

    Returns:
        Objeto Path apuntando al archivo CSV descargado.
    """
    path = kagglehub.dataset_download(dataset_handle)
    print("Path to dataset files:", path)

    dataset_dir = Path(path)
    matches_csv = dataset_dir / "Matches.csv"
    if matches_csv.exists():
        return matches_csv

    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {dataset_dir}")
    return csv_files[0]


def load_and_clean_data(
    filepath: Optional[str | Path] = None,
    division: Optional[str] = None,
    min_year: Optional[int] = 2020,
) -> pd.DataFrame:
    """
    Carga el dataset (descargándolo de Kaggle si no se indica `filepath`),
    estandariza nombres de columnas, limpia valores nulos, normaliza tipos numéricos
    y ordena cronológicamente los partidos.

    Args:
        filepath: Ruta opcional al CSV local. Si es None, descarga desde KaggleHub.
        division: Código de liga opcional para filtrar (ej. 'SP1' LaLiga, 'E0' Premier,
                  o None por defecto para incluir todas las ligas del dataset).
        min_year: Año mínimo a filtrar para entrenar con temporadas recientes (default: 2020).

    Returns:
        DataFrame limpio y ordenado por fecha de forma ascendente.
    """
    if filepath is None or not Path(filepath).exists():
        filepath = download_kaggle_dataset()

    df = pd.read_csv(filepath, low_memory=False)

    # Normalizar nombres de columnas si provienen de "club-football-match-data-2000-2025"
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Filtrar por liga/división solo si el usuario especificó una en particular
    if division is not None and "Division" in df.columns:
        df_div = df[df["Division"] == division]
        if not df_div.empty:
            df = df_div.copy()

    # Eliminar filas sin información esencial del resultado o equipos
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Parseo flexible de fechas (soporta YYYY-MM-DD de Kaggle y DD/MM/YYYY de Football-Data)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    if min_year is not None:
        df = df[df["Date"].dt.year >= min_year].copy()

    # Asegurar formato numérico en goles, estadísticas de juego y cuotas
    numeric_cols = ["FTHG", "FTAG"] + [
        c for c in OPTIONAL_STAT_COLUMNS + ODDS_COLUMNS if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputar estadísticas secundarias faltantes con la mediana
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Filtrar únicamente resultados válidos (H: Local, D: Empate, A: Visitante)
    df = df[df["FTR"].isin(TARGET_MAP.keys())].copy()
    df["Target"] = df["FTR"].map(TARGET_MAP).astype(int)

    # Ordenar cronológicamente para evitar filtración del futuro (data leakage)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_rolling_features(
    df: pd.DataFrame, window: int = 5
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcula promedios móviles históricos (rolling averages) por equipo para goles a favor,
    goles en contra, tiros al arco y puntos obtenidos en los últimos `window` partidos.
    Utiliza `.shift(1)` antes de `.rolling()` para garantizar cero data leakage.

    Args:
        df: DataFrame limpio y ordenado cronológicamente.
        window: Cantidad de partidos previos a considerar en la ventana móvil.

    Returns:
        Tupla con el DataFrame enriquecido (sin filas iniciales incompletas) y la
        lista de nombres de columnas de características (features) generadas.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["MatchID"] = df.index

    # Puntos obtenidos en cada partido según el resultado
    home_pts = np.where(df["FTR"] == "H", 3, np.where(df["FTR"] == "D", 1, 0))
    away_pts = np.where(df["FTR"] == "A", 3, np.where(df["FTR"] == "D", 1, 0))

    has_shots_target = "HST" in df.columns and "AST" in df.columns

    # Vista unificada a nivel equipo-partido para calcular rachas cronológicas
    home_view = pd.DataFrame(
        {
            "MatchID": df["MatchID"],
            "Date": df["Date"],
            "Team": df["HomeTeam"],
            "GoalsFor": df["FTHG"],
            "GoalsAgainst": df["FTAG"],
            "Points": home_pts,
            "ShotsTarget": df["HST"] if has_shots_target else 0.0,
        }
    )

    away_view = pd.DataFrame(
        {
            "MatchID": df["MatchID"],
            "Date": df["Date"],
            "Team": df["AwayTeam"],
            "GoalsFor": df["FTAG"],
            "GoalsAgainst": df["FTHG"],
            "Points": away_pts,
            "ShotsTarget": df["AST"] if has_shots_target else 0.0,
        }
    )

    team_matches = (
        pd.concat([home_view, away_view], ignore_index=True)
        .sort_values(["Date", "MatchID"])
        .reset_index(drop=True)
    )

    # Cálculo de rolling averages desplazando 1 partido hacia atrás (evita data leakage)
    grouped = team_matches.groupby("Team", group_keys=False)
    for col in ["GoalsFor", "GoalsAgainst", "Points", "ShotsTarget"]:
        team_matches[f"Roll_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

    roll_cols = ["Roll_GoalsFor", "Roll_GoalsAgainst", "Roll_Points", "Roll_ShotsTarget"]

    df = df.merge(
        team_matches[["MatchID", "Team"] + roll_cols],
        left_on=["MatchID", "HomeTeam"],
        right_on=["MatchID", "Team"],
        how="left",
    ).drop(columns=["Team"])

    df = df.rename(
        columns={
            "Roll_GoalsFor": "Home_Roll_GF",
            "Roll_GoalsAgainst": "Home_Roll_GA",
            "Roll_Points": "Home_Roll_Pts",
            "Roll_ShotsTarget": "Home_Roll_SOT",
        }
    )

    df = df.merge(
        team_matches[["MatchID", "Team"] + roll_cols],
        left_on=["MatchID", "AwayTeam"],
        right_on=["MatchID", "Team"],
        how="left",
    ).drop(columns=["Team"])

    df = df.rename(
        columns={
            "Roll_GoalsFor": "Away_Roll_GF",
            "Roll_GoalsAgainst": "Away_Roll_GA",
            "Roll_Points": "Away_Roll_Pts",
            "Roll_ShotsTarget": "Away_Roll_SOT",
        }
    )

    # Diferenciales relativos entre local y visitante
    df["Diff_Roll_Pts"] = df["Home_Roll_Pts"] - df["Away_Roll_Pts"]
    df["Diff_Roll_GF"] = df["Home_Roll_GF"] - df["Away_Roll_GF"]

    feature_cols = [
        "Home_Roll_GF",
        "Home_Roll_GA",
        "Home_Roll_Pts",
        "Away_Roll_GF",
        "Away_Roll_GA",
        "Away_Roll_Pts",
        "Diff_Roll_Pts",
        "Diff_Roll_GF",
    ]

    if has_shots_target:
        feature_cols.extend(["Home_Roll_SOT", "Away_Roll_SOT"])

    # Incluir cuotas si están presentes en el dataset
    if all(col in df.columns for col in ODDS_COLUMNS):
        feature_cols.extend(ODDS_COLUMNS)

    # Eliminar primeros partidos de cada equipo donde aún no existen estadísticas previas
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def get_latest_team_stats(df: pd.DataFrame, team: str) -> Dict[str, float]:
    """
    Obtiene las últimas estadísticas móviles y cuotas registradas de un equipo en el dataset.

    Args:
        df: DataFrame procesado con columnas `Home_Roll_*` y `Away_Roll_*`.
        team: Nombre del equipo.

    Returns:
        Diccionario con claves GF, GA, Pts, SOT y cuotas de referencia.
    """
    team_rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if team_rows.empty:
        return {"GF": 1.4, "GA": 1.2, "Pts": 1.5, "SOT": 4.0}

    last_row = team_rows.iloc[-1]
    if last_row["HomeTeam"] == team:
        return {
            "GF": round(float(last_row["Home_Roll_GF"]), 2),
            "GA": round(float(last_row["Home_Roll_GA"]), 2),
            "Pts": round(float(last_row["Home_Roll_Pts"]), 2),
            "SOT": round(float(last_row.get("Home_Roll_SOT", 4.0)), 2),
        }
    return {
        "GF": round(float(last_row["Away_Roll_GF"]), 2),
        "GA": round(float(last_row["Away_Roll_GA"]), 2),
        "Pts": round(float(last_row["Away_Roll_Pts"]), 2),
        "SOT": round(float(last_row.get("Away_Roll_SOT", 4.0)), 2),
    }
