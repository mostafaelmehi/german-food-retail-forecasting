"""SARIMA (Leitfaden Punkte 5–7): Identifikation, Anpassung und Prognose.

Identifikation und Anpassung verwenden ausschließlich die Trainingsdaten.
Die Testdaten werden erst für die Prognosebewertung benutzt.
"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, adfuller, kpss

SAISON = 12


def jahres_streuung(train: pd.Series) -> pd.DataFrame:
    """Mittelwert und Standardabweichung je Kalenderjahr (nur vollständige Jahre).

    Vorgehen wie in der Beispiel-Hausarbeit (Kap. 3): Wächst die Streuung mit
    dem Niveau, ist die Varianz nicht konstant (Vorlesung 2, Folie 32).
    """
    g = train.groupby(train.index.year)
    tab = pd.DataFrame({"mittelwert": g.mean(), "std": g.std(), "monate": g.size()})
    return tab[tab["monate"] == SAISON].drop(columns="monate")


def plot_jahres_streuung(tab: pd.DataFrame):
    """Standardabweichung je Jahr gegen den Jahresmittelwert; r = Korrelation."""
    r = tab["mittelwert"].corr(tab["std"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tab["mittelwert"], tab["std"])
    for jahr, zeile in tab.iterrows():
        if jahr % 5 == 0:
            ax.annotate(str(jahr), (zeile["mittelwert"], zeile["std"]), fontsize=8)
    ax.set_xlabel("Jahresmittelwert")
    ax.set_ylabel("Standardabweichung im Jahr")
    ax.set_title(f"Streuung vs. Niveau, Trainingsdaten (r = {r:.2f})")
    return fig


def boxcox_lambda(train: pd.Series, alpha: float = 0.05):
    """λ per Maximum Likelihood (Schätzung durch Software, Vorlesung 2, Folie 33).

    Gibt λ und das (1 − alpha)-Konfidenzintervall zurück.
    """
    if (train <= 0).any():
        raise ValueError("Box-Cox setzt strikt positive Werte voraus.")
    _, lam, ki = stats.boxcox(train.to_numpy(), alpha=alpha)
    return lam, ki


def differenzieren(s: pd.Series, d: int = 0, D: int = 0, saison: int = SAISON) -> pd.Series:
    """Wendet (1 − B)^d (1 − B^s)^D auf die Reihe an (Vorlesung 3, Folien 37 und 41).

    Pro Differenz gehen am Anfang Werte verloren: insgesamt d + s·D.
    """
    x = s
    for _ in range(D):
        x = x.diff(saison)
    for _ in range(d):
        x = x.diff()
    return x.dropna()


def stationaritaetstests(x: pd.Series):
    """p-Werte von ADF (H0: nicht stationär) und KPSS (H0: stationär).

    KPSS liefert p-Werte nur im Bereich [0,01; 0,10]: ein Randwert bedeutet
    „≤ 0,01“ bzw. „≥ 0,10“.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Hinweise zu Randwerten/Rückgabeformat
        adf_p = adfuller(x, autolag="AIC")[1]
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    return adf_p, kpss_p


def differenzen_tabelle(reihen: dict) -> pd.DataFrame:
    """Je Differenzierungsschritt: Länge, Varianz, ACF bei Lag 1 und 12, p-Werte ADF/KPSS."""
    zeilen = {}
    for name, x in reihen.items():
        adf_p, kpss_p = stationaritaetstests(x)
        r = acf(x, nlags=SAISON)
        zeilen[name] = {
            "n": len(x),
            "Varianz": x.var(),
            "ACF Lag 1": r[1],
            "ACF Lag 12": r[SAISON],
            "ADF p": adf_p,
            "KPSS p": kpss_p,
        }
    return pd.DataFrame(zeilen).T


def plot_differenzen(reihen: dict, lags: int = 36):
    """Je Differenzierungsschritt: Reihe mit Mittelwert (links) und ACF bis Lag 36 (rechts)."""
    fig, axes = plt.subplots(
        len(reihen), 2, figsize=(12, 3 * len(reihen)), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    for (name, x), (ax_r, ax_a) in zip(reihen.items(), axes):
        ax_r.plot(x.index, x, lw=0.8)
        ax_r.axhline(x.mean(), color="tab:red", lw=0.8, ls="--")
        ax_r.set_title(name)
        plot_acf(x, lags=lags, ax=ax_a, title=f"ACF: {name}")
        ax_a.set_xticks(range(0, lags + 1, SAISON))
    fig.tight_layout()
    return fig