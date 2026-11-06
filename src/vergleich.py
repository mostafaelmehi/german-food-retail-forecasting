"""Modellvergleich (Leitfaden Punkt 8): alle Verfahren unter denselben Bedingungen.

Gleicher Split, gleicher Horizont (24 Monate), gleiche Metriken aus src/evaluation.py.
Die Modelle sind in 02_arima.ipynb und 03_holtwinters.ipynb vor dem Test festgelegt
und werden hier nur mit diesen Festlegungen neu geschätzt.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import arima
from src import holtwinters as hw
from src.evaluation import evaluate, rmse, seasonal_naive

# Festlegungen aus den Modell-Notebooks (vor dem Test getroffen)
SARIMA_ORDER = (1, 0, 1)
SARIMA_SAISONAL = (2, 1, 2)


def alle_prognosen(train: pd.Series, horizont: int) -> dict:
    """Prognosen der drei endgültigen Verfahren, alle nur auf den Trainingsdaten geschätzt."""
    hw_res = hw.anpassen(train, gedaempft=True)
    sarima_res = arima.anpassen(train, SARIMA_ORDER, SARIMA_SAISONAL)
    sarima_punkt, _ = arima.prognose(sarima_res, horizont)
    return {
        "Seasonal Naive": seasonal_naive(train, horizont),
        "Holt-Winters (gedämpft)": hw.prognose(hw_res, horizont, name="holt_winters"),
        "SARIMA(1,0,1)(2,1,2)[12]": sarima_punkt,
    }


def vergleichstabelle(test: pd.Series, prognosen: dict) -> pd.DataFrame:
    """RMSE, MAE, MAPE, mittlerer Fehler und RMSE je Testjahr; sortiert nach RMSE."""
    zeilen = {}
    for name, p in prognosen.items():
        zeilen[name] = {
            **evaluate(test, p),
            "mittlerer Fehler (ŷ − y)": float(np.mean(np.asarray(p) - np.asarray(test))),
            "RMSE Jahr 1": rmse(test.iloc[:12], p.iloc[:12]),
            "RMSE Jahr 2": rmse(test.iloc[12:], p.iloc[12:]),
        }
    return pd.DataFrame(zeilen).T.sort_values("RMSE")


def paarvergleich(test: pd.Series, prognosen: dict, a: str, b: str) -> int:
    """In wie vielen Testmonaten hat Verfahren a den kleineren absoluten Fehler als b?"""
    fa = np.abs(np.asarray(prognosen[a]) - np.asarray(test))
    fb = np.abs(np.asarray(prognosen[b]) - np.asarray(test))
    return int((fa < fb).sum())


def plot_vergleich(train, test, prognosen: dict, jahre_zurueck: int = 2):
    """Testwerte und die Prognosen aller Verfahren in einer Grafik."""
    fig, ax = plt.subplots(figsize=(11, 4))
    start = test.index[0] - pd.DateOffset(years=jahre_zurueck)
    ax.plot(train[start:].index, train[start:], color="grey", lw=1, label="Training")
    ax.plot(test.index, test, color="black", lw=1.8, marker="o", ms=3, label="Test (wahre Werte)")
    farben = ["tab:green", "tab:red", "tab:blue"]
    for (name, p), farbe in zip(prognosen.items(), farben):
        ax.plot(p.index, p, color=farbe, lw=1.4, ls="--", label=name)
    ax.axvline(test.index[0], color="grey", ls=":", lw=1)
    ax.set_xlabel("Monat")
    ax.set_ylabel("Index (2021 = 100)")
    ax.set_title("Vergleich der Prognosen auf den 24 Testmonaten")
    ax.legend(loc="upper left", fontsize=8)
    return fig