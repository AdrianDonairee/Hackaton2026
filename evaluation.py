"""
Módulo de evaluación de modelos multiclase y visualización de matrices de confusión.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


CLASS_LABELS = ["Local (H)", "Empate (D)", "Visitante (A)"]


def evaluate_multiclass_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    model_name: str = "Modelo",
) -> Dict[str, Any]:
    """
    Calcula e imprime métricas de clasificación multiclase (accuracy, balanced accuracy
    y classification report detallado por clase).

    Args:
        model: Clasificador entrenado.
        X_test: Conjunto de características de prueba.
        y_test: Etiquetas reales del conjunto de prueba (0=H, 1=D, 2=A).
        model_name: Nombre descriptivo del modelo para mostrar en consola.

    Returns:
        Diccionario con las métricas calculadas y las predicciones generadas.
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_LABELS,
        zero_division=0,
    )

    print("=" * 65)
    print(f"EVALUACIÓN: {model_name}")
    print("=" * 65)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print("\nReporte de Clasificación:")
    print(report)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "classification_report": report,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modelo",
    save_path: Optional[str | Path] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Genera y grafica la matriz de confusión multiclase utilizando matplotlib y seaborn.

    Args:
        y_true: Etiquetas reales (0=H, 1=D, 2=A).
        y_pred: Etiquetas predichas (0=H, 1=D, 2=A).
        model_name: Nombre del modelo para el título del gráfico.
        save_path: Ruta opcional para guardar la imagen en disco.
        show_plot: Si es True, invoca `plt.show()`.

    Returns:
        Objeto Figure de matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusión - {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Resultado Real", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show_plot:
        plt.show()

    return fig

