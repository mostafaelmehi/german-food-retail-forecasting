"""Tests für src/plots.py: laufen mit `pytest` aus dem Projektordner."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # keine Fenster öffnen
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import plots
from src.data import load_umsatz

DATEN = Path(__file__).resolve().parents[1] / "data" / "raw" / "45212-0004_de.csv"


@pytest.fixture(scope="module")
def s():
    return load_umsatz(str(DATEN))


def test_stl_komponenten_ergeben_original(s):
    res = plots.stl_zerlegung(s)
    summe = res.trend + res.seasonal + res.resid
    assert np.allclose(summe, s)


def test_saisonamplitude_nur_volle_jahre(s):
    tab = plots.saisonamplitude(plots.stl_zerlegung(s))
    # 2026 hat nur Jan–Aug und darf nicht vorkommen
    assert 2026 not in tab.index
    assert (tab["amplitude"] > 0).all()


@pytest.mark.parametrize(
    "funktion", [plots.plot_zeitreihe, plots.plot_saisonfigur, plots.plot_lags]
)
def test_grafiken_liefern_figure(s, funktion):
    fig = funktion(s)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
