"""Holt-Winters (Leitfaden Punkt 8): Anpassung auf den Trainingsdaten, Prognose, Grafiken.

Additive Saison mit Periode 12 (Entscheidung aus notebooks/01_eda.ipynb, Abschnitt 5).
Alle Parameter werden von statsmodels auf den Trainingsdaten geschätzt; die Testdaten
werden nur für die Prognosebewertung benutzt.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SAISON = 12

# Namen der statsmodels-Parameter -> Symbole wie in der Vorlesung
PARAMETER = {
    "smoothing_level": "alpha",
    "smoothing_trend": "beta",
    "smoothing_seasonal": "gamma",
    "damping_trend": "phi",
}


def anpassen(train: pd.Series, gedaempft: bool = False):
    """Holt-Winters mit additivem Trend und additiver Saison auf den Trainingsdaten.

    gedaempft=True: gedämpfter Trend (Faktor phi < 1), der Trend flacht in der
    Prognose ab. Startwerte (Niveau, Trend, Saison) werden mitgeschätzt.
    """
    modell = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=gedaempft,
        seasonal="add",
        seasonal_periods=SAISON,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Konvergenzhinweise von statsmodels
        return modell.fit(optimized=True)


def parameter(res) -> pd.Series:
    """Geschätzte Glättungsparameter alpha, beta, gamma (und phi, falls gedämpft)."""
    werte = {sym: res.params[name] for name, sym in PARAMETER.items()}
    return pd.Series(werte).dropna()


def varianten_vergleichen(train: pd.Series) -> pd.DataFrame:
    """Vergleicht ungedämpften und gedämpften Trend nur auf den Trainingsdaten (AIC).

    Kleinerer AIC = bessere Anpassung unter Berücksichtigung der Parameterzahl.
    """
    zeilen = {}
    for name, gedaempft in [("Holt-Winters", False), ("Holt-Winters (gedämpft)", True)]:
        res = anpassen(train, gedaempft)
        zeile = {"AIC": res.aic, "AICc": res.aicc, "SSE (Training)": res.sse}
        zeile.update(parameter(res).to_dict())
        zeilen[name] = zeile
    return pd.DataFrame(zeilen).T


def prognose(res, horizont: int, name: str = "holt_winters") -> pd.Series:
    """Prognose für die nächsten `horizont` Monate nach dem Trainingsende."""
    p = res.forecast(horizont)
    p.name = name
    return p


def plot_prognose(train, test, prognosen: dict, jahre_zurueck: int = 4):
    """Letzte Trainingsjahre, Testwerte und die Prognosen in einer Grafik."""
    fig, ax = plt.subplots(figsize=(11, 4))
    start = test.index[0] - pd.DateOffset(years=jahre_zurueck)
    ax.plot(train[start:].index, train[start:], color="grey", lw=1, label="Training")
    ax.plot(test.index, test, color="black", lw=1.8, marker="o", ms=3, label="Test (wahre Werte)")
    for (label, p), farbe in zip(prognosen.items(), ["tab:red", "tab:blue", "tab:green"]):
        ax.plot(p.index, p, color=farbe, lw=1.5, ls="--", label=label)
    ax.axvline(test.index[0], color="grey", ls=":", lw=1)
    ax.set_xlabel("Monat")
    ax.set_ylabel("Index (2021 = 100)")
    ax.set_title("Prognose der 24 Testmonate")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def fehler_je_monat(test, prognosen: dict) -> pd.DataFrame:
    """Prognosefehler e_t = ŷ_t − y_t je Testmonat (wie in src/evaluation.py)."""
    tab = pd.DataFrame({"wahr": test})
    for label, p in prognosen.items():
        tab[label] = np.asarray(p) - np.asarray(test)
    return tab


def plot_fehler(fehler: pd.DataFrame):
    """Balken der Prognosefehler je Testmonat; > 0 heißt: Prognose zu hoch."""
    spalten = [c for c in fehler.columns if c != "wahr"]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    breite = 0.8 / len(spalten)
    x = np.arange(len(fehler))
    for i, (c, farbe) in enumerate(zip(spalten, ["tab:red", "tab:blue", "tab:green"])):
        ax.bar(x + (i - (len(spalten) - 1) / 2) * breite, fehler[c], breite, label=c, color=farbe)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:%m/%y}" for d in fehler.index], rotation=90, fontsize=8)
    ax.set_ylabel("Fehler ŷ − y")
    ax.set_title("Prognosefehler je Testmonat")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
