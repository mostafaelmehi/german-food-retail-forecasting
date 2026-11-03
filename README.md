# German Food Retail Forecasting

Time series analysis and forecasting of monthly German food retail sales (Destatis) using SARIMA and Holt-Winters.

Gruppenprojekt im Modul **Zeitreihenanalyse**, Fachhochschule Dortmund (Vortrag 1).

## Fragestellung

Lässt sich der monatliche reale Umsatz im deutschen Lebensmitteleinzelhandel mit ARIMA-Modellen zuverlässig prognostizieren, und wie schneiden diese im Vergleich zu Holt-Winters und einer Seasonal-Naive-Baseline ab?
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

## Methodische Entscheidungen

Gelten für alle Modelle und sind in `src/evaluation.py` umgesetzt. Kein Notebook teilt die Daten anders auf.

| Entscheidung | Festlegung | Begründung |
|---|---|---|
| Validierung | Historischer Train-Test-Split: Training 01/1994–08/2024 (368 Monate), Test 09/2024–08/2026 (24 Monate) | Zeitliche statt zufälliger Aufteilung (Vorlesung 2, Folien 25 und 27); 24 Monate = zwei volle Saisonzyklen |
| Prognose | Einmaliges Training, Prognose aller 24 Testmonate auf einmal | Entspricht der Planungssituation; Testmenge wird erst nach der Modellwahl verwendet |
| Hauptmetrik | RMSE, ergänzend MAE und MAPE; Fehler e_t = ŷ_t − y_t | RMSE und MAE: Vorlesung 2, Folie 24; RMSE gewichtet große Fehler (z. B. Dezember) stärker; MAPE zusätzlich, da die Reihe keine Nullen enthält |
| Baseline | Seasonal Naive: ŷ_{T+h} = Wert desselben Monats im letzten Trainingsjahr | Einfaches Referenzverfahren als Maßstab für SARIMA und Holt-Winters |

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
