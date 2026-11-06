"""SARIMA (Leitfaden Punkte 5–7): Identifikation, Anpassung und Prognose.

Identifikation und Anpassung verwenden ausschließlich die Trainingsdaten.
Die Testdaten werden erst für die Prognosebewertung benutzt.
"""
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

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

def plot_acf_pacf(x: pd.Series, lags: int = 36, titel: str = ""):
    """ACF (links) und PACF (rechts) bis Lag 36, damit die Lags 12, 24, 36 sichtbar sind."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    plot_acf(x, lags=lags, ax=ax1, title=f"ACF {titel}")
    plot_pacf(x, lags=lags, ax=ax2, method="ywm", title=f"PACF {titel}")
    for ax in (ax1, ax2):
        ax.set_xticks(range(0, lags + 1, SAISON))
    fig.tight_layout()
    return fig


def acf_pacf_tabelle(x: pd.Series, lags: int = 36, alpha: float = 0.05) -> pd.DataFrame:
    """ACF und PACF je Lag; signifikant = außerhalb des 95%-Bands (wie in den Grafiken)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # Hinweis zum Rückgabeformat
        a, a_ki = acf(x, nlags=lags, alpha=alpha)
        p, p_ki = pacf(x, nlags=lags, alpha=alpha, method="ywm")
    tab = pd.DataFrame(
        {
            "ACF": a,
            "ACF signifikant": (a_ki[:, 0] > 0) | (a_ki[:, 1] < 0),
            "PACF": p,
            "PACF signifikant": (p_ki[:, 0] > 0) | (p_ki[:, 1] < 0),
        },
        index=pd.RangeIndex(0, lags + 1, name="Lag"),
    )
    return tab.iloc[1:]

def modell_name(order, seasonal_order) -> str:
    """Kurzname wie SARIMA(1,0,1)(2,1,2)[12]."""
    p, d, q = order
    P, D, Q = seasonal_order
    return f"SARIMA({p},{d},{q})({P},{D},{Q})[{SAISON}]"


def anpassen(train: pd.Series, order, seasonal_order, trend: str = "c"):
    """Schätzt ein SARIMA-Modell per Maximum Likelihood auf den Trainingsdaten.

    order = (p, d, q), seasonal_order = (P, D, Q); die Saisonlänge ist 12.
    trend="c": konstanter Term in der differenzierten Gleichung.
    """
    modell = SARIMAX(train, order=order, seasonal_order=(*seasonal_order, SAISON), trend=trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Konvergenz wird in aic_gitter über mle_retvals geprüft
        return modell.fit(disp=False, maxiter=200)


def aic_gitter(train: pd.Series, d: int, D: int, trend: str = "c", max_ordnung: int = 2) -> pd.DataFrame:
    """Alle Modelle mit p, q, P, Q ∈ {0, …, max_ordnung} bei festem d, D und trend.

    AIC ist nur bei gleichem d, D und gleichem trend vergleichbar (gleiche Daten
    in der Likelihood). Sortiert nach AIC; ΔAIC = Abstand zum besten Modell.
    """
    zeilen = []
    for p, q, P, Q in itertools.product(range(max_ordnung + 1), repeat=4):
        res = anpassen(train, (p, d, q), (P, D, Q), trend)
        zeilen.append(
            {
                "Modell": modell_name((p, d, q), (P, D, Q)),
                "p": p, "q": q, "P": P, "Q": Q,
                "Parameter": len(res.params),
                "AIC": res.aic,
                "konvergiert": bool(res.mle_retvals.get("converged", True)),
            }
        )
    tab = pd.DataFrame(zeilen).sort_values("AIC").reset_index(drop=True)
    tab["ΔAIC"] = tab["AIC"] - tab["AIC"].min()
    return tab
# Symbole wie im Bericht: φ/θ (nicht-saisonal), Φ/Θ (saisonal), c (Konstante)
SYMBOLE = {
    "intercept": "c",
    "ar.L1": "φ₁", "ar.L2": "φ₂", "ar.L3": "φ₃",
    "ma.L1": "θ₁", "ma.L2": "θ₂",
    "ar.S.L12": "Φ₁", "ar.S.L24": "Φ₂",
    "ma.S.L12": "Θ₁", "ma.S.L24": "Θ₂",
    "sigma2": "σ²",
}


def koeffizienten(res) -> pd.DataFrame:
    """Geschätzte Koeffizienten mit Standardfehler und p-Wert (Vorlesung 3, Folie 34)."""
    tab = pd.DataFrame(
        {"Schätzer": res.params, "Std.-Fehler": res.bse, "p-Wert": res.pvalues}
    )
    tab.index = [SYMBOLE.get(name, name) for name in tab.index]
    return tab


def residuen(res) -> pd.Series:
    """Residuen ohne die ersten d + 12·D Werte (durch die Differenzen nicht aussagekräftig)."""
    return res.resid.iloc[res.loglikelihood_burn:]


def ljung_box(res, lags=(12, 24)) -> pd.DataFrame:
    """Ljung-Box-Test der Residuen (H0: keine Autokorrelation bis Lag k).

    Freiheitsgrade um die Zahl der ARMA-Parameter p + q + P + Q reduziert.
    """
    m = res.model
    k = m.k_ar_params + m.k_ma_params + m.k_seasonal_ar_params + m.k_seasonal_ma_params
    tab = acorr_ljungbox(residuen(res), lags=list(lags), model_df=k)
    tab.index.name = "Lag"
    return tab.rename(columns={"lb_stat": "Q", "lb_pvalue": "p-Wert"})


def kandidaten_vergleich(modelle: dict) -> pd.DataFrame:
    """Kennzahlen je Kandidat für die Modellwahl in Abschnitt 6 (alles Trainingsdaten)."""
    zeilen = {}
    for name, res in modelle.items():
        koef = koeffizienten(res).drop(index=["c", "σ²"], errors="ignore")
        lb = ljung_box(res)
        e = residuen(res)
        zeilen[name] = {
            "Parameter": len(res.params),
            "AIC": res.aic,
            "σ²": res.params["sigma2"],
            "nicht signifikante Koeffizienten (p > 0,05)": int((koef["p-Wert"] > 0.05).sum()),
            "Ljung-Box p (Lag 12)": lb.loc[12, "p-Wert"],
            "Ljung-Box p (Lag 24)": lb.loc[24, "p-Wert"],
            "Jarque-Bera p": stats.jarque_bera(e).pvalue,
        }
    tab = pd.DataFrame(zeilen).T
    tab["ΔAIC"] = tab["AIC"] - tab["AIC"].min()
    return tab


def plot_residuen(res, titel: str = "", lags: int = 36):
    """Residuendiagnose: Verlauf, ACF, Histogramm mit Normalverteilung, Q-Q-Plot."""
    e = residuen(res)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.plot(e.index, e, lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Residuen im Zeitverlauf")
    plot_acf(e, lags=lags, ax=axes[0, 1], title="ACF der Residuen")
    axes[0, 1].set_xticks(range(0, lags + 1, SAISON))
    ax = axes[1, 0]
    ax.hist(e, bins=30, density=True, alpha=0.7)
    x = np.linspace(e.min(), e.max(), 200)
    ax.plot(x, stats.norm.pdf(x, e.mean(), e.std()), color="tab:red", label="Normalverteilung")
    ax.set_title("Histogramm")
    ax.legend(fontsize=8)
    stats.probplot(e, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q-Plot")
    fig.suptitle(titel)
    fig.tight_layout()
    return fig
def _zahl(x: float) -> str:
    """Betrag mit drei Nachkommastellen und Dezimalkomma (LaTeX)."""
    return f"{abs(x):.3f}".replace(".", "{,}")


def _polynom(res, praefix: str, ar: bool) -> str:
    """AR-Polynom (1 − a₁B^k − …) oder MA-Polynom (1 + b₁B^k + …) aus den Schätzern."""
    teile = ["1"]
    for name, wert in res.params.items():
        if name.startswith(praefix):
            lag = int(name.split("L")[-1])
            koef = -wert if ar else wert
            zeichen = "+" if koef >= 0 else "-"
            potenz = "B" if lag == 1 else f"B^{{{lag}}}"
            teile.append(f"{zeichen} {_zahl(koef)}\\,{potenz}")
    return f"({' '.join(teile)})" if len(teile) > 1 else ""


def modellgleichung(res) -> str:
    """Geschätzte Modellgleichung in Backshift-Schreibweise als LaTeX (Vorlesung 3, Folien 34 und 37)."""
    m = res.model
    links = _polynom(res, "ar.L", ar=True) + _polynom(res, "ar.S.L", ar=True)
    if m.k_diff:
        links += "(1 - B)"
    if m.k_seasonal_diff:
        links += f"(1 - B^{{{m.seasonal_periods}}})"
    rechts = _polynom(res, "ma.L", ar=False) + _polynom(res, "ma.S.L", ar=False)
    c = res.params.get("intercept")
    konstante = "" if c is None else f"{'-' if c < 0 else ''}{_zahl(c)} + "
    return f"{links}\\,Y_t = {konstante}{rechts}\\,\\varepsilon_t"
def prognose(res, horizont: int, alpha: float = 0.05):
    """Prognose für die nächsten `horizont` Monate nach dem Trainingsende.

    Gibt die Punktprognose (Series) und das (1 − alpha)-Prognoseintervall
    (DataFrame mit Spalten "unten", "oben") zurück. Das Modell wird dabei nicht
    neu geschätzt; Testwerte gehen nicht ein (Vorlesung 3, Folie 42).
    """
    fc = res.get_forecast(steps=horizont)
    punkt = fc.predicted_mean.rename("sarima")
    ki = fc.conf_int(alpha=alpha)
    ki.columns = ["unten", "oben"]
    return punkt, ki


def abdeckung(test: pd.Series, intervall: pd.DataFrame) -> float:
    """Anteil der Testwerte, die im Prognoseintervall liegen."""
    drin = (test >= intervall["unten"]) & (test <= intervall["oben"])
    return float(drin.mean())


def plot_prognose(train, test, punkt, intervall, jahre_zurueck: int = 4):
    """Letzte Trainingsjahre, Testwerte, SARIMA-Prognose und 95%-Prognoseintervall."""
    fig, ax = plt.subplots(figsize=(11, 4))
    start = test.index[0] - pd.DateOffset(years=jahre_zurueck)
    ax.plot(train[start:].index, train[start:], color="grey", lw=1, label="Training")
    ax.plot(test.index, test, color="black", lw=1.8, marker="o", ms=3, label="Test (wahre Werte)")
    ax.plot(punkt.index, punkt, color="tab:blue", lw=1.5, ls="--", label="SARIMA-Prognose")
    ax.fill_between(intervall.index, intervall["unten"], intervall["oben"],
                    color="tab:blue", alpha=0.15, label="95%-Prognoseintervall")
    ax.axvline(test.index[0], color="grey", ls=":", lw=1)
    ax.set_xlabel("Monat")
    ax.set_ylabel("Index (2021 = 100)")
    ax.set_title("SARIMA: Prognose der 24 Testmonate")
    ax.legend(loc="upper left", fontsize=8)
    return fig