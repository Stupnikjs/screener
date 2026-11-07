"""
Cache disque persistant pour le scanner : cours de clôture et PER.

Disposition (dans le dossier racine, par défaut `.cache` à côté du CSV) :
    prices/px_<ticker>.csv   un fichier par titre : colonnes date, close (cours ajustés)
    prices_meta.json         par titre : date du dernier téléchargement, début couvert, statut
    pe.json                  PER (12 mois / prévu) avec date de téléchargement

Principes :
- un titre « à jour » (téléchargé depuis moins de `ttl_hours`) ne génère aucune requête ;
- un titre plus ancien est complété : on ne retélécharge que les derniers jours (+ chevauchement) ;
- les cours sont ajustés des dividendes/splits : si le chevauchement ne colle plus avec le
  cache, tout l'historique du titre est retéléchargé ;
- les titres sans aucun cours sont mémorisés comme « morts » pendant `dead_ttl_hours`, mais
  seulement si l'échec est rare (sinon c'est probablement le réseau, pas le titre) ;
- l'écriture se fait par lots : un scan interrompu garde ce qui a déjà été téléchargé.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

# network(tickers, start, end, progress_cb) -> DataFrame de cours (colonnes = tickers)
Network = Callable[[list, str, str, "Callable[[float], None] | None"], pd.DataFrame]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DiskCache:
    META = "prices_meta.json"
    PE = "pe.json"

    def __init__(self, root, ttl_hours: float = 12.0, dead_ttl_hours: float = 24.0,
                 pe_ttl_hours: float = 24.0, overlap_days: int = 7, refresh: bool = False,
                 batch: int = 200, tolerance: float = 1e-3, dead_max_share: float = 0.15,
                 now: Callable[[], datetime] | None = None):
        self.root = Path(root)
        self.ttl = ttl_hours
        self.dead_ttl = dead_ttl_hours
        self.pe_ttl = pe_ttl_hours
        self.overlap = pd.Timedelta(days=overlap_days)
        self.refresh = refresh
        self.batch = batch
        self.tolerance = tolerance          # écart relatif toléré sur le chevauchement
        self.dead_max_share = dead_max_share
        self._now = now or datetime.now
        self.stats: dict[str, int] = {}
        self._pe: dict | None = None

    # ------------------------------------------------------------------ fichiers
    def _series_path(self, ticker: str) -> Path:
        # préfixe : « CON », « NUL »... sont des noms réservés sous Windows
        return self.root / "prices" / f"px_{_UNSAFE.sub('_', ticker)}.csv"

    def _load_meta(self) -> dict:
        try:
            data = json.loads((self.root / self.META).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_meta(self, meta: dict) -> None:
        _atomic_write(self.root / self.META, json.dumps(meta, indent=0, sort_keys=True))

    def _load_series(self, ticker: str) -> pd.Series | None:
        try:
            df = pd.read_csv(self._series_path(ticker), index_col=0, parse_dates=True)
            s = df["close"].dropna()
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return None
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return None
        s = s.sort_index().astype(float)
        s.index.name = None  # même index qu'un téléchargement direct
        return s

    def _save_series(self, ticker: str, s: pd.Series) -> None:
        _atomic_write(self._series_path(ticker),
                      s.rename("close").to_csv(index_label="date"))

    # -------------------------------------------------------------------- cours
    def fetch_prices(self, tickers: list, start: str, end: str, network: Network,
                     progress_cb: Callable[[float], None] | None = None) -> pd.DataFrame:
        """Cours de clôture des `tickers` sur [start, end[, depuis le disque quand c'est possible."""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        now = self._now()
        stamp = now.isoformat(timespec="seconds")
        meta = self._load_meta()
        self.stats = dict.fromkeys(("fresh", "updated", "full", "dead", "redo", "missing"), 0)

        series: dict[str, pd.Series] = {}
        full: list[str] = []
        incr: list[str] = []
        for t in tickers:
            m = meta.get(t)
            if self.refresh or not isinstance(m, dict):
                full.append(t)
                continue
            try:
                age = (now - datetime.fromisoformat(m["fetched_at"])).total_seconds() / 3600
                if m.get("status") == "empty":
                    if age < self.dead_ttl:
                        self.stats["dead"] += 1
                    else:
                        full.append(t)
                    continue
                if pd.Timestamp(m["covered_from"]) > start_ts:
                    full.append(t)          # fenêtre demandée plus longue que celle en cache
                    continue
            except (KeyError, ValueError, TypeError):
                full.append(t)
                continue
            s = self._load_series(t)
            if s is None:
                full.append(t)
                continue
            series[t] = s
            if age < self.ttl:
                self.stats["fresh"] += 1
            else:
                incr.append(t)

        total = max(1, len(tickers))
        state = {"done": len(tickers) - len(full) - len(incr)}
        if progress_cb and state["done"]:
            progress_cb(state["done"] / total)

        def call(part: list, g_start: str) -> pd.DataFrame:
            base = state["done"]
            cb = (lambda f: progress_cb((base + f * len(part)) / total)) if progress_cb else None
            df = network(part, g_start, end, cb)
            state["done"] += len(part)
            return df if df is not None else pd.DataFrame()

        def store_full(part: list, df: pd.DataFrame) -> list[str]:
            """Enregistre les titres reçus ; retourne ceux qui n'ont rien renvoyé."""
            missing = []
            for t in part:
                s = df[t].dropna() if t in df.columns else pd.Series(dtype=float)
                if s.empty:
                    missing.append(t)
                    continue
                self._save_series(t, s)
                meta[t] = {"status": "ok", "fetched_at": stamp,
                           "covered_from": start_ts.strftime("%Y-%m-%d")}
                series[t] = s
            return missing

        # 1) titres inconnus / à recharger en entier, par lots (progression sauvegardée)
        no_data: list[str] = []
        for i in range(0, len(full), self.batch):
            part = full[i:i + self.batch]
            no_data += store_full(part, call(part, start))
            self.stats["full"] += len(part)
            self._save_meta(meta)

        # 2) titres en cache mais périmés : seulement les derniers jours
        redo: list[str] = []
        for i in range(0, len(incr), self.batch):
            part = incr[i:i + self.batch]
            g_start = (min(series[t].index[-1] for t in part) - self.overlap).strftime("%Y-%m-%d")
            df = call(part, g_start)
            for t in part:
                new = df[t].dropna() if t in df.columns else pd.Series(dtype=float)
                if new.empty:
                    continue  # échec ou rien de neuf : on garde le cache, retenté au prochain run
                old = series[t]
                common = old.index.intersection(new.index)
                if common.empty or ((new[common] / old[common].replace(0, np.nan) - 1)
                                    .abs().max() > self.tolerance):
                    redo.append(t)   # ajustement (dividende, split) ou trou : tout retélécharger
                    continue
                merged = pd.concat([old[old.index < new.index.min()], new])
                self._save_series(t, merged)
                series[t] = merged
                meta[t] = {**meta[t], "fetched_at": stamp}
                self.stats["updated"] += 1
            self._save_meta(meta)

        # 3) historiques à refaire : on ne remplace le cache que si le téléchargement réussit
        for i in range(0, len(redo), self.batch):
            part = redo[i:i + self.batch]
            store_full(part, call(part, start))
            self.stats["redo"] += len(part)
            self._save_meta(meta)

        # 4) titres sans cours : mémorisés « morts » seulement si l'échec est rare
        self.stats["missing"] = len(no_data)
        never_ok = [t for t in no_data if meta.get(t, {}).get("status") != "ok"]
        if full and len(no_data) <= self.dead_max_share * len(full):
            for t in never_ok:
                meta[t] = {"status": "empty", "fetched_at": stamp}
            if never_ok:
                self._save_meta(meta)

        cols = [t for t in tickers if t in series]
        if not cols:
            return pd.DataFrame()
        df = pd.concat({t: series[t] for t in cols}, axis=1).sort_index()
        return df[(df.index >= start_ts) & (df.index < end_ts)]

    # ---------------------------------------------------------------------- PER
    def _pe_data(self) -> dict:
        if self._pe is None:
            try:
                data = json.loads((self.root / self.PE).read_text(encoding="utf-8"))
                self._pe = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._pe = {}
        return self._pe

    def get_pe(self, ticker: str) -> tuple[float, float] | None:
        """(PER 12 mois, PER prévu) en cache et encore valable, sinon None."""
        if self.refresh:
            return None
        e = self._pe_data().get(ticker)
        try:
            age = (self._now() - datetime.fromisoformat(e["fetched_at"])).total_seconds() / 3600
            if age >= self.pe_ttl:
                return None
            return (np.nan if e["trailing"] is None else float(e["trailing"]),
                    np.nan if e["forward"] is None else float(e["forward"]))
        except (KeyError, ValueError, TypeError):
            return None

    def save_pe(self, items: dict) -> None:
        """Enregistre {ticker: (PER, PER prévu)} ; NaN devient null."""
        if not items:
            return
        data = self._pe_data()
        stamp = self._now().isoformat(timespec="seconds")
        for t, (tr, fw) in items.items():
            data[t] = {"trailing": None if pd.isna(tr) else float(tr),
                       "forward": None if pd.isna(fw) else float(fw),
                       "fetched_at": stamp}
        _atomic_write(self.root / self.PE, json.dumps(data, indent=0, sort_keys=True))