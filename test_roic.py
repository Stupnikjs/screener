"""
Tests du filtre ROIC (calcul, cache disque, filtre, branchement CLI). Aucun accès réseau.

    python -m unittest -v test_roic
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import yfinance  # noqa: F401
except ImportError:
    _stub = types.ModuleType("yfinance")
    _stub.download = lambda *a, **k: pd.DataFrame()
    _stub.Ticker = object
    sys.modules["yfinance"] = _stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache as cache_mod  # noqa: E402
import cli  # noqa: E402
import scanner as ds  # noqa: E402

YEARS = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])  # ordre Yahoo : récent d'abord


def income(ebit, tax=None):
    rows = {"Operating Income": ebit}
    if tax is not None:
        rows["Tax Rate For Calcs"] = tax
    return pd.DataFrame(rows, index=YEARS[:len(ebit)]).T


def balance(**rows):
    n = len(next(iter(rows.values())))
    return pd.DataFrame(rows, index=YEARS[:n]).T


class FakeClock:
    def __init__(self, t="2026-02-20 10:00"):
        self.t = datetime.fromisoformat(t)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class TestComputeRoic(unittest.TestCase):
    def test_uses_invested_capital_and_tax_rate(self):
        inc = income([200.0, 100.0], tax=[0.25, 0.20])
        bal = balance(**{"Invested Capital": [1000.0, 800.0]})
        last, avg = ds.compute_roic(inc, bal)
        self.assertAlmostEqual(last, 200 * 0.75 / 1000 * 100)          # 15 %
        self.assertAlmostEqual(avg, (15.0 + 100 * 0.8 / 800 * 100) / 2)  # (15 + 10) / 2

    def test_fallback_invested_capital_from_debt_equity_cash(self):
        inc = income([300.0], tax=[0.0])
        bal = balance(**{"Total Debt": [500.0], "Stockholders Equity": [700.0],
                         "Cash And Cash Equivalents": [200.0]})
        last, _ = ds.compute_roic(inc, bal)
        self.assertAlmostEqual(last, 300 / 1000 * 100)

    def test_missing_tax_uses_default_and_absurd_tax_is_replaced(self):
        bal = balance(**{"Invested Capital": [1000.0]})
        a, _ = ds.compute_roic(income([100.0]), bal)
        b, _ = ds.compute_roic(income([100.0], tax=[0.95]), bal)      # taux aberrant -> repli
        expected = 100 * (1 - ds.ROIC_DEFAULT_TAX) / 1000 * 100
        self.assertAlmostEqual(a, expected)
        self.assertAlmostEqual(b, expected)

    def test_non_positive_invested_capital_is_ignored(self):
        inc = income([100.0, 100.0], tax=[0.0, 0.0])
        bal = balance(**{"Invested Capital": [500.0, -50.0]})
        last, avg = ds.compute_roic(inc, bal)
        self.assertAlmostEqual(last, 20.0)
        self.assertAlmostEqual(avg, 20.0)                              # l'exercice négatif est exclu

    def test_only_negative_capital_gives_nan(self):
        last, avg = ds.compute_roic(income([100.0], tax=[0.0]),
                                    balance(**{"Invested Capital": [-10.0]}))
        self.assertTrue(np.isnan(last) and np.isnan(avg))

    def test_missing_rows_give_nan(self):
        self.assertTrue(np.isnan(ds.compute_roic(pd.DataFrame(), pd.DataFrame())[0]))
        no_ebit = pd.DataFrame({"Net Income": [1.0]}, index=YEARS[:1]).T
        self.assertTrue(np.isnan(ds.compute_roic(no_ebit, balance(**{"Invested Capital": [1.0]}))[0]))
        self.assertTrue(np.isnan(ds.compute_roic(income([1.0]), balance(**{"Total Debt": [1.0]}))[0]))

    def test_last_is_most_recent_year_whatever_the_column_order(self):
        inc = income([300.0, 100.0, 100.0], tax=[0.0, 0.0, 0.0])
        bal = balance(**{"Invested Capital": [1000.0, 1000.0, 1000.0]})
        last, avg = ds.compute_roic(inc, bal)
        self.assertAlmostEqual(last, 30.0)
        self.assertAlmostEqual(avg, (30 + 10 + 10) / 3)
        # même résultat si Yahoo renvoie les colonnes dans l'autre sens
        last2, avg2 = ds.compute_roic(inc[inc.columns[::-1]], bal[bal.columns[::-1]])
        self.assertAlmostEqual(last2, last)
        self.assertAlmostEqual(avg2, avg)

    def test_average_is_limited_to_recent_years(self):
        n = ds.ROIC_MAX_YEARS + 2
        idx = pd.date_range(end="2025-12-31", periods=n, freq="YE")
        inc = pd.DataFrame({"Operating Income": [1000.0] + [100.0] * (n - 1)}, index=idx[::-1]).T
        bal = pd.DataFrame({"Invested Capital": [1000.0] * n}, index=idx[::-1]).T
        inc.loc["Tax Rate For Calcs"] = 0.0
        _, avg = ds.compute_roic(inc, bal)
        self.assertAlmostEqual(avg, (100 + 10 * (ds.ROIC_MAX_YEARS - 1)) / ds.ROIC_MAX_YEARS)


class TestGetRoicCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(ds.configure_cache, None)
        ds._ROIC_CACHE.clear()
        ds._ROIC_NEW.clear()
        self.addCleanup(ds._ROIC_CACHE.clear)
        self.addCleanup(ds._ROIC_NEW.clear)
        self.tk = mock.Mock(income_stmt=income([200.0], tax=[0.0]),
                            balance_sheet=balance(**{"Invested Capital": [1000.0]}))

    def test_reads_then_writes_disk_cache(self):
        ds.configure_cache(self.dir.name)
        with mock.patch.object(ds.yf, "Ticker", return_value=self.tk) as tk:
            self.assertEqual(ds.get_roic("A"), (20.0, 20.0))
            ds.add_roic(pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]}))  # persiste
            ds._ROIC_CACHE.clear()                                             # nouvelle « session »
            self.assertEqual(ds.get_roic("A"), (20.0, 20.0))
            self.assertEqual(tk.call_count, 1)                                 # 2e lecture : disque
        self.assertTrue((Path(self.dir.name) / "roic.json").exists())

    def test_failed_request_is_never_persisted(self):
        ds.configure_cache(self.dir.name)
        with mock.patch.object(ds.yf, "Ticker", side_effect=ConnectionError("down")):
            r = ds.get_roic("A")
            ds.add_roic(pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]}))
        self.assertTrue(np.isnan(r[0]))
        self.assertFalse((Path(self.dir.name) / "roic.json").exists())

    def test_empty_statements_are_not_persisted(self):
        ds.configure_cache(self.dir.name)
        empty = mock.Mock(income_stmt=pd.DataFrame(), balance_sheet=pd.DataFrame())
        with mock.patch.object(ds.yf, "Ticker", return_value=empty):
            self.assertTrue(np.isnan(ds.get_roic("A")[0]))
            ds.add_roic(pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]}))
        self.assertFalse((Path(self.dir.name) / "roic.json").exists())

    def test_etf_is_not_queried(self):
        with mock.patch.object(ds.yf, "Ticker", return_value=self.tk) as tk:
            out = ds.add_roic(pd.DataFrame({"yahoo": ["E", "A"], "section": ["ETF", "Stock"]}))
        self.assertEqual(tk.call_count, 1)
        self.assertTrue(np.isnan(out.loc[out["yahoo"] == "E", "ROIC %"].iloc[0]))
        self.assertEqual(out.loc[out["yahoo"] == "A", "ROIC moy %"].iloc[0], 20.0)

    def test_add_roic_without_stocks_does_not_crash(self):
        out = ds.add_roic(pd.DataFrame({"yahoo": ["E"], "section": ["ETF"]}))
        self.assertTrue(out["ROIC %"].isna().all())


class TestDiskCacheRoic(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.clock = FakeClock()

    def mk(self, **kw):
        return cache_mod.DiskCache(self.dir.name, now=self.clock, **kw)

    def test_roundtrip_with_nan(self):
        self.mk().save_roic({"A": (12.5, 10.0), "B": (np.nan, np.nan)})
        got = self.mk().get_roic("A")
        self.assertEqual(got, (12.5, 10.0))
        b = self.mk().get_roic("B")
        self.assertTrue(np.isnan(b[0]) and np.isnan(b[1]))

    def test_expiry_unknown_and_refresh(self):
        c = self.mk(roic_ttl_hours=24)
        c.save_roic({"A": (12.5, 10.0)})
        self.assertIsNone(c.get_roic("ZZZ"))
        self.clock.advance(hours=23)
        self.assertEqual(self.mk(roic_ttl_hours=24).get_roic("A"), (12.5, 10.0))
        self.clock.advance(hours=2)
        self.assertIsNone(self.mk(roic_ttl_hours=24).get_roic("A"))
        self.assertIsNone(self.mk(refresh=True).get_roic("A"))

    def test_default_ttl_is_a_week(self):
        self.mk().save_roic({"A": (1.0, 1.0)})
        self.clock.advance(days=6)
        self.assertIsNotNone(self.mk().get_roic("A"))
        self.clock.advance(days=2)
        self.assertIsNone(self.mk().get_roic("A"))

    def test_corrupt_file_is_ignored(self):
        (Path(self.dir.name) / "roic.json").write_text("{pas du json", encoding="utf-8")
        self.assertIsNone(self.mk().get_roic("A"))


class TestFilterRoic(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "yahoo": ["HI", "LO", "NA", "ETF"],
            "section": ["Stock", "Stock", "Stock", "ETF"],
            "ROIC moy %": [15.0, 5.0, np.nan, np.nan],
            "ROIC %": [3.0, 20.0, np.nan, np.nan],
        })

    def test_threshold_and_etf_passes(self):
        out = ds.filter_roic(self.df, 10, "ROIC moy %", False)
        self.assertEqual(out["yahoo"].tolist(), ["HI", "ETF"])

    def test_keep_missing(self):
        out = ds.filter_roic(self.df, 10, "ROIC moy %", True)
        self.assertEqual(out["yahoo"].tolist(), ["HI", "NA", "ETF"])

    def test_last_year_column(self):
        out = ds.filter_roic(self.df, 10, "ROIC %", False)
        self.assertEqual(out["yahoo"].tolist(), ["LO", "ETF"])


class TestCliRoic(unittest.TestCase):
    ROIC = {"AAA.PA": (8.0, 14.0), "BBB.PA": (30.0, 4.0)}

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(ds.configure_cache, None)
        ds._ROIC_CACHE.clear()
        ds._ROIC_NEW.clear()
        self.addCleanup(ds._ROIC_CACHE.clear)
        self.addCleanup(ds._ROIC_NEW.clear)
        self.csv = os.path.join(self.dir.name, "u.csv")
        pd.DataFrame({"ticker": ["AAA.FR", "BBB.FR"], "desc": ["Aaa", "Bbb"],
                      "section": ["Stock", "Stock"], "currency": ["EUR", "EUR"]}
                     ).to_csv(self.csv, index=False)
        today = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(today - pd.Timedelta(days=500), today)
        n = len(idx)
        decline = np.concatenate([np.full(n - 40, 100.0), np.linspace(100, 40, 40)])
        self.frame = pd.DataFrame({"AAA.PA": decline, "BBB.PA": decline * 2}, index=idx)

    def run_cli(self, *args):
        def fake_chunk(tickers, start, end):
            cols = [t for t in tickers if t in self.frame.columns]
            return self.frame.loc[start:end, cols]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["cli.py", *args, "--csv", self.csv]), \
                mock.patch.object(ds, "fetch_chunk", fake_chunk), \
                mock.patch.object(ds, "get_roic", lambda y: self.ROIC.get(y, (np.nan, np.nan))), \
                mock.patch.object(ds.time, "sleep", lambda s: None), \
                redirect_stdout(out), redirect_stderr(err):
            code = cli.main()
        return code, out.getvalue(), err.getvalue()

    def names(self, out):
        return [line.split()[0] for line in out.strip().splitlines()]

    def test_without_roic_flag_nothing_is_filtered(self):
        _, out, _ = self.run_cli("40", "3m", "--no-cache")
        self.assertEqual(self.names(out), ["AAA.FR", "BBB.FR"])

    def test_default_uses_average_roic(self):
        _, out, _ = self.run_cli("40", "3m", "--no-cache", "--roic-min", "10")
        self.assertEqual(self.names(out), ["AAA.FR"])          # moyenne 14 >= 10 ; BBB : moyenne 4

    def test_roic_type_last(self):
        _, out, _ = self.run_cli("40", "3m", "--no-cache", "--roic-min", "10", "--roic-type", "last")
        self.assertEqual(self.names(out), ["BBB.FR"])          # dernier exercice 30 >= 10 ; AAA : 8

    def test_nothing_matches(self):
        code, out, _ = self.run_cli("40", "3m", "--no-cache", "--roic-min", "50")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "")

    def test_show_roic_adds_columns_without_filtering(self):
        _, out, _ = self.run_cli("40", "3m", "--no-cache", "--show-roic", "-v")
        self.assertIn("ROIC %", out)
        self.assertIn("ROIC moy %", out)
        self.assertIn("AAA.FR", out)
        self.assertIn("BBB.FR", out)

    def test_keep_no_roic(self):
        ds_roic = {"AAA.PA": (8.0, 14.0)}                      # BBB sans ROIC
        with mock.patch.object(type(self), "ROIC", ds_roic):
            _, out, _ = self.run_cli("40", "3m", "--no-cache", "--roic-min", "10")
            self.assertEqual(self.names(out), ["AAA.FR"])
            _, out, _ = self.run_cli("40", "3m", "--no-cache", "--roic-min", "10", "--keep-no-roic")
            self.assertEqual(self.names(out), ["AAA.FR", "BBB.FR"])


if __name__ == "__main__":
    unittest.main()