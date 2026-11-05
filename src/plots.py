"""Deskriptive Analyse (Leitfaden Punkt 3): Grafiken, STL-Zerlegung, Saisonamplitude.

Alle Funktionen geben die Matplotlib-Figure zurück, damit das Notebook sie anzeigen
und mit `speichern()` nach figures/ schreiben kann.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

FIGURES = Path(__file__).resolve().parents[1] / "figures"

MONATSNAMEN = [
    "Jan",
    "Feb",
    "Mär",
    "Apr",
    "Mai",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Okt",
    "Nov",
    "Dez",
]


def speichern(fig, name: str) -> Path:
    """Speichert eine Grafik als figures/<name>.png (z. B. name='02_saisonfigur')."""
    FIGURES.mkdir(exist_ok=True)
    pfad = FIGURES / f"{name}.png"
    fig.savefig(pfad, dpi=150, bbox_inches="tight")
    return pfad


def plot_zeitreihe(s: pd.Series, fenster: int = 12):
    """Zeitreihe mit gleitendem Durchschnitt SMA_k (Vorlesung 2: SMA_t = 1/k · Σ y_{t-j}).

    Mit k = 12 mittelt jedes Fenster genau ein Jahr, die Saison fällt heraus
    und der Trend wird sichtbar. Markiert werden nur Maximum und Minimum des
    SMA, beide aus der Reihe berechnet.
    """
    sma = s.rolling(fenster).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s, lw=0.8, color="tab:blue", label="Umsatzindex (real, 2021=100)")
    ax.plot(sma.index, sma, lw=2, color="tab:red", label=f"SMA (k={fenster})")
    for datum, wert, versatz in [
        (sma.idxmax(), sma.max(), 10),
        (sma.idxmin(), sma.min(), -14),
    ]:
        ax.plot(datum, wert, "o", color="black", ms=5, zorder=3)
        ax.annotate(
            f"{datum:%m/%Y}: {wert:.1f}",
            (datum, wert),
            xytext=(0, versatz),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Monat")
    ax.set_ylabel("Index (2021 = 100)")
    ax.set_title("Realer Umsatz im Lebensmitteleinzelhandel, Deutschland")
    ax.legend(loc="lower right")
    return fig


def plot_saisonfigur(s: pd.Series):
    """Boxplot je Kalendermonat: Lage und Streuung der Saisonfigur."""
    monate = [s[s.index.month == m].to_numpy() for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(monate, tick_labels=MONATSNAMEN)
    ax.plot(
        range(1, 13),
        [m.mean() for m in monate],
        "o-",
        color="tab:red",
        label="Mittelwert",
    )
    ax.set_ylabel("Index (2021 = 100)")
    ax.set_title("Saisonfigur: Verteilung je Kalendermonat")
    ax.legend()
    return fig


def plot_lags(s: pd.Series, lags=(1, 12)):
    """Lag-Plots (Vorlesung 1): jeder Punkt ist (y_{t-k}, y_t); r = Korrelation."""
    fig, axes = plt.subplots(1, len(lags), figsize=(4.5 * len(lags), 4))
    for ax, k in zip(np.atleast_1d(axes), lags):
        x, y = s.shift(k), s
        r = x.corr(y)
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel(f"$y_{{t-{k}}}$")
        ax.set_ylabel("$y_t$")
        ax.set_title(f"Lag {k}  (r = {r:.2f})")
    fig.tight_layout()
    return fig


def stl_zerlegung(s: pd.Series, periode: int = 12, robust: bool = True):
    """STL-Zerlegung y_t = T_t + S_t + R_t (Vorlesung 1: nur zur Beschreibung, nicht zur Prognose).

    robust=True, damit Ausreißer den Trend und die Saison weniger verzerren
    und stattdessen im Rest landen.
    """
    return STL(s, period=periode, robust=robust).fit()


def plot_stl(res):
    """Vier Panels: Original, Trend, Saison, Rest."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    teile = [
        (res.observed, "Original"),
        (res.trend, "Trend $T_t$"),
        (res.seasonal, "Saison $S_t$"),
        (res.resid, "Rest $R_t$"),
    ]
    for ax, (reihe, titel) in zip(axes, teile):
        if titel.startswith("Rest"):
            ax.scatter(reihe.index, reihe, s=4)
            ax.axhline(0, color="black", lw=0.6)
        else:
            ax.plot(reihe.index, reihe, lw=0.9)
        ax.set_ylabel(titel)
    axes[0].set_title("STL-Zerlegung (Periode 12, robust)")
    fig.tight_layout()
    return fig


def saisonamplitude(res) -> pd.DataFrame:
    """Je Jahr: Spannweite der Saisonkomponente und mittleres Trendniveau.

    Wächst die Amplitude mit dem Niveau → multiplikativ; bleibt sie etwa
    konstant → additiv. Nur vollständige Jahre.
    """
    df = pd.DataFrame({"saison": res.seasonal, "trend": res.trend})
    g = df.groupby(df.index.year)
    tab = pd.DataFrame(
        {
            "amplitude": g["saison"].max() - g["saison"].min(),
            "niveau": g["trend"].mean(),
            "monate": g.size(),
        }
    )
    return tab[tab["monate"] == 12].drop(columns="monate")


def plot_saisonamplitude(tab: pd.DataFrame):
    """Amplitude je Jahr gegen das Trendniveau; r nahe 0 spricht für additiv."""
    r = tab["amplitude"].corr(tab["niveau"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tab["niveau"], tab["amplitude"])
    for jahr, zeile in tab.iterrows():
        if jahr % 5 == 0:
            ax.annotate(str(jahr), (zeile["niveau"], zeile["amplitude"]), fontsize=8)
    ax.set_xlabel("Mittleres Trendniveau im Jahr")
    ax.set_ylabel("Saisonamplitude (max − min)")
    ax.set_title(f"Saisonamplitude vs. Niveau  (r = {r:.2f})")
    return fig
