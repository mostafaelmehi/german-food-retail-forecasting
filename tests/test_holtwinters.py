"""Tests für src/holtwinters.py: laufen mit `pytest` aus dem Projektordner."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src import holtwinters as hw
from src.data import load_umsatz
from src.evaluation import train_test_split

DATEN = Path(__file__).resolve().parents[1] / "data" / "raw" / "45212-0004_de.csv"


@pytest.fixture(scope="module")
def split():
    return train_test_split(load_umsatz(str(DATEN)))


@pytest.fixture(scope="module")
def res(split):
    train, _ = split
    return hw.anpassen(train, gedaempft=True)


def test_nur_trainingsdaten(split, res):
    train, _ = split
    # Das Modell kennt genau die Trainingsmonate, keinen Testmonat
    assert len(res.fittedvalues) == len(train)
    assert res.fittedvalues.index[-1] == train.index[-1]


def test_prognose_passt_zum_test(split, res):
    _, test = split
    p = hw.prognose(res, len(test))
    assert p.index.equals(test.index)


def test_parameter_im_zulaessigen_bereich(res):
    par = hw.parameter(res)
    assert set(par.index) == {"alpha", "beta", "gamma", "phi"}
    assert ((par >= 0) & (par <= 1)).all()


def test_vergleich_zwei_varianten(split):
    train, _ = split
    tab = hw.varianten_vergleichen(train)
    assert len(tab) == 2
    assert tab["AIC"].notna().all()


def test_fehler_vorzeichen(split, res):
    _, test = split
    p = hw.prognose(res, len(test))
    tab = hw.fehler_je_monat(test, {"HW": p})
    # e_t = ŷ_t − y_t
    assert (tab["HW"] == p.to_numpy() - test.to_numpy()).all()
    plt.close(hw.plot_fehler(tab))
