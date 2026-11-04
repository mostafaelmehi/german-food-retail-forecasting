"""Gemeinsame Evaluation: zeitlicher Train/Test-Split, Fehlermaße und Seasonal-Naive-Baseline."""
import numpy as np
import pandas as pd

SAISON = 12
TEST_MONATE = 24


def train_test_split(s: pd.Series, test_monate: int = TEST_MONATE):
    """Zeitlicher (nicht zufälliger) Split: die letzten test_monate Werte bilden die Testmenge."""
    if not s.index.is_monotonic_increasing:
        raise ValueError("Der Zeitindex ist nicht aufsteigend sortiert.")
    if s.isna().any():
        raise ValueError("Die Zeitreihe enthält fehlende Werte.")

    train = s.iloc[:-test_monate]
    test = s.iloc[-test_monate:]
    return train, test

def _prognosefehler(y_true, y_pred):
    """Prognosefehler e_t = ŷ_t − y_t (Vorlesung 2, Folie 24)."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("Wahre Werte und Prognosen haben unterschiedliche Zeitindizes.")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Wahre Werte und Prognosen haben unterschiedliche Längen.")
    return y_pred - y_true


def mae(y_true, y_pred):
    """Mean Absolute Error: (1/n) · Σ |e_t|"""
    e = _prognosefehler(y_true, y_pred)
    return float(np.mean(np.abs(e)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error: √[(1/n) · Σ e_t²] (Hauptmetrik)"""
    e = _prognosefehler(y_true, y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error in Prozent: (100/n) · Σ |e_t / y_t|"""
    e = _prognosefehler(y_true, y_pred)
    y = np.asarray(y_true, dtype=float)
    return float(100 * np.mean(np.abs(e / y)))



def evaluate(y_true, y_pred):
    """Alle drei Metriken auf einmal, z. B. für die Vergleichstabelle."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
    }


def seasonal_naive(train, horizont=TEST_MONATE, saison=SAISON):
    """Seasonal Naive: Prognose = Wert desselben Monats im letzten Trainingsjahr."""
    letzte_saison = train.iloc[-saison:].to_numpy()
    werte = np.resize(letzte_saison, horizont)
    index = pd.date_range(train.index[-1], periods=horizont + 1, freq=train.index.freq)[1:]
    return pd.Series(werte, index=index, name="seasonal_naive")