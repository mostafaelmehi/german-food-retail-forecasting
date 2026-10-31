"""Laden und Bereinigen der Destatis-Tabelle 45212-0004 (Umsatzindex Einzelhandel)."""
import pandas as pd

MONATE = {"Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5, "Juni": 6,
          "Juli": 7, "August": 8, "September": 9, "Oktober": 10, "November": 11, "Dezember": 12}


def load_umsatz(path: str) -> pd.Series:
    """Liest die GENESIS-CSV und gibt eine monatliche Zeitreihe (Umsatzindex, 2021=100) zurück."""
    df = pd.read_csv(
        path, sep=";", encoding="utf-8-sig", header=None, skiprows=9,
        names=["wz_code", "wz_name", "jahr", "monat", "index", "veraenderung"],
        dtype=str,
    )
    # Nur echte Datenzeilen (Fußzeile mit Copyright usw. entfernen)
    df = df[df["wz_code"].str.startswith("WZ08", na=False)]
    # "..." = Angabe fällt später an -> weglassen
    df = df[df["index"] != "..."]
    df["index"] = df["index"].str.replace(",", ".", regex=False).astype(float)
    df["datum"] = pd.to_datetime(
        dict(year=df["jahr"].astype(int), month=df["monat"].map(MONATE), day=1))
    s = df.set_index("datum")["index"].sort_index()
    s.index.freq = "MS"
    s.name = "umsatzindex_real"
    return s
