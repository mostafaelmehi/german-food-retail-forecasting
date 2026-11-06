"""Tests für src/vergleich.py: laufen mit `pytest` aus dem Projektordner."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src import vergleich
from src.data import load_umsatz
from src.evaluation import train_test_split

DATEN = Path(__file__).resolve().parents[1] / "data" / "raw" / "45212-0004_de.csv"


@pytest.fixture(scope="module")
def daten():
    train, test = train_test_split(load_umsatz(str(DATEN)))
    return train, test, vergleich.alle_prognosen(train, len(test))


def test_gleiche_bedingungen(daten):
    _, test, prognosen = daten
    # drei Verfahren, alle mit genau den 24 Testmonaten als Index
    assert len(prognosen) == 3
    for p in prognosen.values():
        assert p.index.equals(test.index)


def test_tabelle_und_grafik(daten):
    train, test, prognosen = daten
    tab = vergleich.vergleichstabelle(test, prognosen)
    assert list(tab["RMSE"]) == sorted(tab["RMSE"])
    namen = list(prognosen)
    n = vergleich.paarvergleich(test, prognosen, namen[1], namen[2])
    assert 0 <= n <= len(test)
    plt.close(vergleich.plot_vergleich(train, test, prognosen))