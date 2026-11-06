"""Tests für src/arima.py: laufen mit `pytest` aus dem Projektordner."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src import arima
from src.data import load_umsatz
from src.evaluation import train_test_split

DATEN = Path(__file__).resolve().parents[1] / "data" / "raw" / "45212-0004_de.csv"


@pytest.fixture(scope="module")
def train():
    train, _ = train_test_split(load_umsatz(str(DATEN)))
    return train


def test_jahres_streuung_nur_volle_jahre(train):
    tab = arima.jahres_streuung(train)
    # 1994–2023 vollständig, 2024 hat im Training nur 8 Monate
    assert tab.index[0] == 1994 and tab.index[-1] == 2023
    assert len(tab) == 30
    plt.close(arima.plot_jahres_streuung(tab))


def test_boxcox_lehnt_nichtpositive_werte_ab(train):
    kaputt = train.copy()
    kaputt.iloc[0] = 0
    with pytest.raises(ValueError):
        arima.boxcox_lambda(kaputt)


def test_differenzieren_laenge_und_wert(train):
    x = arima.differenzieren(train, d=1, D=1)
    # d + 12·D = 13 Werte gehen verloren
    assert len(x) == len(train) - 13
    # (1 − B)(1 − B^12) Y_t = Y_t − Y_{t−1} − Y_{t−12} + Y_{t−13}
    t = x.index[0]
    y = train
    erwartet = y[t] - y.shift(1)[t] - y.shift(12)[t] + y.shift(13)[t]
    assert x[t] == pytest.approx(erwartet)


def test_tabelle_p_werte_im_bereich(train):
    reihen = {"Original": train, "saisonal": arima.differenzieren(train, D=1)}
    tab = arima.differenzen_tabelle(reihen)
    assert list(tab.index) == ["Original", "saisonal"]
    assert ((tab[["ADF p", "KPSS p"]] >= 0) & (tab[["ADF p", "KPSS p"]] <= 1)).all().all()
    plt.close(arima.plot_differenzen(reihen))