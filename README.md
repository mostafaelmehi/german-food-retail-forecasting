# German Food Retail Forecasting

Time series analysis and forecasting of monthly German food retail sales (Destatis) using SARIMA and Holt-Winters.

Gruppenprojekt im Modul **Zeitreihenanalyse**, Fachhochschule Dortmund (Vortrag 1).

## Fragestellung

Lässt sich der monatliche reale Umsatz im deutschen Lebensmitteleinzelhandel mit ARIMA-Modellen zuverlässig prognostizieren, und wie schneiden diese im Vergleich zu Holt-Winters ab?

## Daten

| | |
|---|---|
| Quelle | Statistisches Bundesamt (Destatis), GENESIS-Online |
| Tabelle | 45212-0004 – Umsatzindex im Einzelhandel |
| Wirtschaftszweig | WZ08-4711-02 – Einzelhandel mit Lebensmitteln |
| Preisart | in konstanten Preisen (real), 2021 = 100 |
| Zeitraum | Januar 1994 – August 2026 (392 Monate) |
| Frequenz | monatlich, Originalwerte (nicht saisonbereinigt) |

Lizenz der Daten: Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0).

## Projektstruktur

```
├── data/raw/        Originaldaten von Destatis (werden nie verändert)
├── src/             Gemeinsamer Code (Daten laden, Split, Metriken)
├── notebooks/       Analysen (EDA, ARIMA, Holt-Winters, Vergleich)
└── figures/         Grafiken für Bericht und Präsentation
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install             # entfernt Notebook-Outputs vor jedem Commit
```

## Daten laden

```python
from src.data import load_umsatz
s = load_umsatz("data/raw/45212-0004_de.csv")
```

## Workflow

- Kein direkter Push auf `main`.
- Pro Aufgabe ein Branch (`feature/eda`, `feature/arima`, ...).
- Merge nur über Pull Request mit Review durch das andere Teammitglied.

## Team

- _Name 1_
- _Name 2_

## Lizenz

Code: MIT (siehe `LICENSE`). Daten: © Statistisches Bundesamt (Destatis), dl-de/by-2-0.
