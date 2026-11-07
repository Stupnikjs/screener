"""
Tests du cache disque (cache.py) et de son branchement dans scanner / cli.
Aucun accès réseau.

    python -m unittest -v test_cache
"""
from __future__ import annotations

import io
import json
import logging
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

START, END = "2026-01-01", "2026-03-01"


def bdays(first: str, last: str) -> pd.DatetimeIndex:
    return pd.bdate_range(first, last)


class FakeMarket:
    """Réseau simulé : une série par ticker, tronquée à `today`, avec journal des appels."""

    def __init__(self, tickers, today="2026-02-20", dead=(), factor=None):
        self.tickers = set(tickers)
        self.dead = set(dead)
        self.today = pd.Timestamp(today)
        self.factor = factor or {}   # ticker -> facteur d'ajustement appliqué à tout l'historique
        self.calls: list[tuple] = []
        self.fail: set[str] = set()  # tickers qui « échouent » silencieusement

    def price(self, t, d):
        base = 100 + (sum(map(ord, t)) % 50)
        return (base + (d - pd.Timestamp("2026-01-01")).days * 0.1) * self.factor.get(t, 1.0)

    def __call__(self, tickers, start, end, cb=None):
        self.calls.append((tuple(tickers), start, end))
        out = {}
        for t in tickers:
            if t in self.dead or t in self.fail or t not in self.tickers:
                continue
            idx = bdays(start, min(pd.Timestamp(end) - pd.Timedelta(days=1), self.today))
            out[t] = pd.Series([self.price(t, d) for d in idx], index=idx)
        if cb:
            cb(1.0)
        return pd.DataFrame(out) if out else pd.DataFrame()


class Clock:
    def __init__(self, t="2026-02-20 10:00"):
        self.t = datetime.fromisoformat(t)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class CacheCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.clock = Clock()
        self.mk = lambda **kw: cache_mod.DiskCache(self.dir.name, now=self.clock, **kw)


class TestPrices(CacheCase):
    def test_first_fetch_then_fresh_cache_makes_no_request(self):
        net = FakeMarket(["A", "B"])
        out1 = self.mk().fetch_prices(["A", "B"], START, END, net)
        self.assertEqual(len(net.calls), 1)
        self.clock.advance(hours=1)
        c = self.mk()
        out2 = c.fetch_prices(["A", "B"], START, END, net)
        self.assertEqual(len(net.calls), 1)          # aucun nouvel appel
        self.assertEqual(c.stats["fresh"], 2)
        pd.testing.assert_frame_equal(out1, out2, check_freq=False)

    def test_stale_cache_only_fetches_recent_days_and_appends(self):
        net = FakeMarket(["A"], today="2026-02-13")
        self.mk().fetch_prices(["A"], START, END, net)
        net.today = pd.Timestamp("2026-02-20")
        self.clock.advance(hours=13)
        c = self.mk()
        out = c.fetch_prices(["A"], START, END, net)
        _, g_start, _ = net.calls[-1]
        self.assertEqual(g_start, "2026-02-06")      # dernier cours (13/02) - 7 jours
        self.assertEqual(out.index.max(), pd.Timestamp("2026-02-20"))
        self.assertEqual(c.stats["updated"], 1)
        expected = pd.Series([net.price("A", d) for d in bdays(START, "2026-02-20")],
                             index=bdays(START, "2026-02-20"))
        pd.testing.assert_series_equal(out["A"], expected, check_names=False, check_freq=False)
        self.assertFalse(out.index.duplicated().any())

    def test_adjustment_mismatch_triggers_full_refetch(self):
        net = FakeMarket(["A"], today="2026-02-13")
        self.mk().fetch_prices(["A"], START, END, net)
        net.today = pd.Timestamp("2026-02-20")
        net.factor["A"] = 0.95                       # dividende : tout l'historique est ré-ajusté
        self.clock.advance(hours=13)
        c = self.mk()
        out = c.fetch_prices(["A"], START, END, net)
        self.assertEqual(c.stats["redo"], 1)
        self.assertEqual(net.calls[-1][1], START)    # historique complet
        self.assertAlmostEqual(out["A"].iloc[0], net.price("A", out.index[0]))

    def test_failed_redo_keeps_old_cache(self):
        net = FakeMarket(["A"], today="2026-02-13")
        self.mk().fetch_prices(["A"], START, END, net)
        net.today = pd.Timestamp("2026-02-20")
        net.factor["A"] = 0.9
        self.clock.advance(hours=13)

        real = net.__call__
        n = {"i": 0}

        def flaky(tickers, start, end, cb=None):
            n["i"] += 1
            return real(tickers, start, end, cb) if n["i"] == 1 else pd.DataFrame()
        out = self.mk().fetch_prices(["A"], START, END, flaky)
        self.assertIn("A", out.columns)              # cache ancien conservé
        self.assertEqual(out.index.max(), pd.Timestamp("2026-02-13"))

    def test_longer_window_forces_full_fetch(self):
        net = FakeMarket(["A"])
        self.mk().fetch_prices(["A"], "2026-01-15", END, net)
        self.mk().fetch_prices(["A"], START, END, net)
        self.assertEqual(net.calls[-1][1], START)
        self.assertEqual(len(net.calls), 2)

    def test_shorter_or_later_window_is_served_from_cache(self):
        net = FakeMarket(["A"])
        self.mk().fetch_prices(["A"], START, END, net)
        out = self.mk().fetch_prices(["A"], "2026-02-01", END, net)
        self.assertEqual(len(net.calls), 1)
        self.assertGreaterEqual(out.index.min(), pd.Timestamp("2026-02-01"))

    def test_failed_update_keeps_cache_and_is_retried_next_time(self):
        net = FakeMarket(["A"], today="2026-02-13")
        self.mk().fetch_prices(["A"], START, END, net)
        net.today = pd.Timestamp("2026-02-20")
        net.fail.add("A")
        self.clock.advance(hours=13)
        out = self.mk().fetch_prices(["A"], START, END, net)
        self.assertEqual(out.index.max(), pd.Timestamp("2026-02-13"))   # données anciennes servies
        net.fail.clear()
        c = self.mk()
        out = c.fetch_prices(["A"], START, END, net)                     # retenté tout de suite
        self.assertEqual(out.index.max(), pd.Timestamp("2026-02-20"))
        self.assertEqual(c.stats["updated"], 1)

    def test_dead_ticker_is_remembered_in_a_healthy_run(self):
        names = [f"T{i}" for i in range(20)]
        net = FakeMarket(names, dead={"T0"})
        c = self.mk()
        c.fetch_prices(names, START, END, net)
        self.assertEqual(c.stats["missing"], 1)
        self.clock.advance(hours=1)
        calls = len(net.calls)
        c = self.mk()
        c.fetch_prices(names, START, END, net)
        self.assertEqual(len(net.calls), calls)      # T0 n'est pas redemandé
        self.assertEqual(c.stats["dead"], 1)

    def test_dead_marker_expires(self):
        names = [f"T{i}" for i in range(20)]
        net = FakeMarket(names, dead={"T0"})
        self.mk(dead_ttl_hours=24).fetch_prices(names, START, END, net)
        self.clock.advance(hours=25)
        calls = len(net.calls)
        self.mk(dead_ttl_hours=24).fetch_prices(names, START, END, net)
        self.assertEqual(net.calls[calls][0], ("T0",))   # retenté après expiration

    def test_mass_failure_is_not_remembered_as_dead(self):
        names = [f"T{i}" for i in range(20)]
        net = FakeMarket(names)
        net.fail = set(names[:10])                    # panne réseau : la moitié ne répond pas
        self.mk().fetch_prices(names, START, END, net)
        net.fail.clear()
        c = self.mk()
        out = c.fetch_prices(names, START, END, net)
        self.assertEqual(c.stats["dead"], 0)
        self.assertEqual(len(out.columns), 20)        # tout est revenu au run suivant

    def test_refresh_ignores_cache_and_rewrites_it(self):
        net = FakeMarket(["A"])
        self.mk().fetch_prices(["A"], START, END, net)
        self.mk(refresh=True).fetch_prices(["A"], START, END, net)
        self.assertEqual(len(net.calls), 2)
        c = self.mk()
        c.fetch_prices(["A"], START, END, net)
        self.assertEqual(len(net.calls), 2)           # le cache réécrit sert à nouveau

    def test_corrupt_series_file_is_refetched(self):
        net = FakeMarket(["A"])
        c = self.mk()
        c.fetch_prices(["A"], START, END, net)
        c._series_path("A").write_text("n'importe quoi,,\n\x00", encoding="utf-8")
        out = self.mk().fetch_prices(["A"], START, END, net)
        self.assertEqual(len(net.calls), 2)
        self.assertIn("A", out.columns)

    def test_corrupt_meta_is_ignored(self):
        net = FakeMarket(["A"])
        c = self.mk()
        c.fetch_prices(["A"], START, END, net)
        (Path(self.dir.name) / cache_mod.DiskCache.META).write_text("{pas du json", encoding="utf-8")
        self.mk().fetch_prices(["A"], START, END, net)
        self.assertEqual(len(net.calls), 2)

    def test_interrupted_scan_keeps_finished_batches(self):
        names = [f"T{i}" for i in range(10)]
        net = FakeMarket(names)
        real = net.__call__
        n = {"i": 0}

        def interrupted(tickers, start, end, cb=None):
            n["i"] += 1
            if n["i"] == 3:
                raise KeyboardInterrupt
            return real(tickers, start, end, cb)
        with self.assertRaises(KeyboardInterrupt):
            self.mk(batch=3).fetch_prices(names, START, END, interrupted)
        net.calls.clear()
        c = self.mk(batch=3)
        out = c.fetch_prices(names, START, END, net)
        self.assertEqual(c.stats["fresh"], 6)          # 2 lots de 3 déjà sur disque
        self.assertEqual(sum(len(call[0]) for call in net.calls), 4)
        self.assertEqual(len(out.columns), 10)

    def test_windows_reserved_names_and_symbols_are_safe(self):
        c = self.mk()
        for t in ("CON", "NUL", "^FCHI", "A/B", "BRK-B", "NOVO-B.CO"):
            self.assertTrue(c._series_path(t).name.startswith("px_"))
            self.assertNotIn("/", c._series_path(t).name)
        net = FakeMarket(["CON", "^FCHI"])
        out = self.mk().fetch_prices(["CON", "^FCHI"], START, END, net)
        self.assertEqual(sorted(out.columns), ["CON", "^FCHI"])

    def test_progress_reaches_one_and_is_monotonic(self):
        names = [f"T{i}" for i in range(7)]
        net = FakeMarket(names)
        seen = []
        self.mk(batch=3).fetch_prices(names, START, END, net, progress_cb=seen.append)
        self.assertAlmostEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_empty_result_returns_empty_frame(self):
        net = FakeMarket([])
        self.assertTrue(self.mk().fetch_prices(["A"], START, END, net).empty)


class TestPeCache(CacheCase):
    def test_roundtrip_with_nan(self):
        c = self.mk()
        c.save_pe({"A": (12.5, 10.0), "ETF": (np.nan, np.nan)})
        got = self.mk().get_pe("A")
        self.assertEqual(got, (12.5, 10.0))
        etf = self.mk().get_pe("ETF")
        self.assertTrue(np.isnan(etf[0]) and np.isnan(etf[1]))
        raw = json.loads((Path(self.dir.name) / "pe.json").read_text(encoding="utf-8"))
        self.assertIsNone(raw["ETF"]["trailing"])     # JSON valide, pas de NaN

    def test_expiry_unknown_and_refresh(self):
        c = self.mk()
        c.save_pe({"A": (12.5, 10.0)})
        self.assertIsNone(c.get_pe("ZZZ"))
        self.clock.advance(hours=25)
        self.assertIsNone(self.mk().get_pe("A"))
        self.clock.advance(hours=-25)
        self.assertIsNone(self.mk(refresh=True).get_pe("A"))


class TestScannerIntegration(CacheCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(ds.configure_cache, None)
        ds._PE_CACHE.clear()
        ds._PE_NEW.clear()
        self.addCleanup(ds._PE_CACHE.clear)
        self.addCleanup(ds._PE_NEW.clear)

    def test_fetch_prices_uses_cache_when_configured(self):
        net = FakeMarket(["A", "B"])
        ds.configure_cache(self.dir.name)
        with mock.patch.object(ds, "fetch_chunk", lambda t, s, e: net(list(t), s, e)):
            ds.fetch_prices(["A", "B"], START, END, retry_rounds=0)
            n = len(net.calls)
            out = ds.fetch_prices(["A", "B"], START, END, retry_rounds=0)
        self.assertEqual(len(net.calls), n)
        self.assertEqual(sorted(out.columns), ["A", "B"])

    def test_without_cache_every_call_hits_the_network(self):
        net = FakeMarket(["A"])
        ds.configure_cache(None)
        with mock.patch.object(ds, "fetch_chunk", lambda t, s, e: net(list(t), s, e)):
            ds.fetch_prices(["A"], START, END, retry_rounds=0)
            ds.fetch_prices(["A"], START, END, retry_rounds=0)
        self.assertEqual(len(net.calls), 2)

    def test_get_pe_reads_then_writes_disk_cache(self):
        ds.configure_cache(self.dir.name)
        info = {"trailingPE": 11.0, "forwardPE": 9.0}
        with mock.patch.object(ds.yf, "Ticker", return_value=mock.Mock(info=info)) as tk:
            self.assertEqual(ds.get_pe("A"), (11.0, 9.0))
            df = pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]})
            ds.add_pe(df)                              # persiste ce qui vient du réseau
            ds._PE_CACHE.clear()                       # nouvelle « session »
            self.assertEqual(ds.get_pe("A"), (11.0, 9.0))
            self.assertEqual(tk.call_count, 1)         # 2e lecture : disque, pas Yahoo

    def test_failed_pe_request_is_never_persisted(self):
        ds.configure_cache(self.dir.name)
        with mock.patch.object(ds.yf, "Ticker", side_effect=ConnectionError("down")):
            r = ds.get_pe("A")
            ds.add_pe(pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]}))
        self.assertTrue(np.isnan(r[0]))
        self.assertFalse((Path(self.dir.name) / "pe.json").exists())

    def test_empty_pe_info_is_not_persisted(self):
        ds.configure_cache(self.dir.name)
        with mock.patch.object(ds.yf, "Ticker", return_value=mock.Mock(info={})):
            ds.get_pe("A")
            ds.add_pe(pd.DataFrame({"yahoo": ["A"], "section": ["Stock"]}))
        self.assertFalse((Path(self.dir.name) / "pe.json").exists())


class TestCli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(ds.configure_cache, None)
        self.csv = os.path.join(self.dir.name, "u.csv")
        pd.DataFrame({"ticker": ["AAA.FR", "BBB.FR"], "desc": ["Aaa", "Bbb"],
                      "section": ["Stock", "Stock"], "currency": ["EUR", "EUR"]}
                     ).to_csv(self.csv, index=False)
        today = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(today - pd.Timedelta(days=500), today)
        n = len(idx)                                      # stable puis -60 % sur les 40 derniers jours
        decline = np.concatenate([np.full(n - 40, 100.0), np.linspace(100, 40, 40)])
        self.frame = pd.DataFrame({"AAA.PA": decline, "BBB.PA": decline * 2}, index=idx)
        self.calls = 0

    def run_cli(self, *args):
        def fake_chunk(tickers, start, end):
            self.calls += 1
            cols = [t for t in tickers if t in self.frame.columns]
            return self.frame.loc[start:end, cols]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["cli.py", *args, "--csv", self.csv]), \
                mock.patch.object(ds, "fetch_chunk", fake_chunk), \
                mock.patch.object(ds.time, "sleep", lambda s: None), \
                redirect_stdout(out), redirect_stderr(err):
            code = cli.main()
        return code, out.getvalue(), err.getvalue()

    def test_default_creates_cache_next_to_csv_and_second_run_needs_no_download(self):
        code, out, err = self.run_cli("40", "3m", "-v")
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isdir(os.path.join(self.dir.name, ds.CACHE_DIR, "prices")))
        first = self.calls
        self.assertGreater(first, 0)
        _, out2, err2 = self.run_cli("40", "3m", "-v")
        self.assertEqual(self.calls, first)               # aucune requête de plus
        self.assertIn("2 à jour", err2)
        self.assertEqual(out, out2)                       # même résultat

    def test_no_cache_flag(self):
        self.run_cli("40", "3m", "--no-cache")
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, ds.CACHE_DIR)))
        first = self.calls
        self.run_cli("40", "3m", "--no-cache")
        self.assertEqual(self.calls, 2 * first)

    def test_refresh_flag_redownloads(self):
        self.run_cli("40", "3m")
        first = self.calls
        self.run_cli("40", "3m", "--refresh")
        self.assertEqual(self.calls, 2 * first)

    def test_cache_dir_option(self):
        custom = os.path.join(self.dir.name, "elsewhere")
        self.run_cli("40", "3m", "--cache-dir", custom)
        self.assertTrue(os.path.isdir(os.path.join(custom, "prices")))
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, ds.CACHE_DIR)))

    def test_global_cache_is_reset_after_main(self):
        self.run_cli("40", "3m")
        self.assertIsNone(ds.CACHE)

    def test_cache_dir_is_excluded_from_llm_export(self):
        root = Path(self.dir.name)
        (root / ds.CACHE_DIR / "prices").mkdir(parents=True)
        (root / ds.CACHE_DIR / "prices" / "px_A.csv").write_text("date,close\n", encoding="utf-8")
        (root / "m.py").write_text("x = 1\n", encoding="utf-8")
        out = cli.dump_for_llm(root).read_text(encoding="utf-8")
        self.assertIn("===== m.py =====", out)
        self.assertNotIn("px_A", out)


class TestQuietOutput(TestCli):
    """Par défaut : seulement ticker et nom ; -v rétablit tout le reste."""

    def test_default_output_is_only_ticker_and_name(self):
        code, out, err = self.run_cli("40", "3m")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)                      # une ligne par titre, rien d'autre
        self.assertEqual(sorted(l.split()[0] for l in lines), ["AAA.FR", "BBB.FR"])
        names = {l.split()[0]: l.split(None, 1)[1] for l in lines}
        self.assertEqual(names, {"AAA.FR": "Aaa", "BBB.FR": "Bbb"})
        self.assertEqual(err, "")                            # ni progression, ni bilan, ni astuce

    def test_columns_are_aligned_and_multiword_names_kept(self):
        pd.DataFrame({"ticker": ["AAA.FR", "BBBBB.FR"], "desc": ["Aaa", "Bbb Group SA"],
                      "section": ["Stock", "Stock"], "currency": ["EUR", "EUR"]}
                     ).to_csv(self.csv, index=False)
        self.frame = self.frame.rename(columns={"BBB.PA": "BBBBB.PA"})
        _, out, _ = self.run_cli("40", "3m")
        rows = {l.split()[0]: l for l in out.splitlines()}
        self.assertTrue(rows["AAA.FR"].startswith("AAA.FR    Aaa"))   # nom aligné sur la plus longue
        self.assertTrue(rows["BBBBB.FR"].endswith("Bbb Group SA"))

    def test_nothing_printed_when_no_match(self):
        code, out, err = self.run_cli("99", "3m")
        self.assertEqual((code, out, err), (0, "", ""))

    def test_verbose_restores_details(self):
        _, out, err = self.run_cli("40", "3m", "-v")
        self.assertIn("titres retenus sur", out)
        self.assertIn("Dernier cours", out)
        self.assertIn("Téléchargement", err)
        self.assertIn("Cache :", err)

    def test_out_csv_still_written_silently(self):
        target = os.path.join(self.dir.name, "res.csv")
        _, out, err = self.run_cli("40", "3m", "--out", target)
        self.assertEqual(sorted(l.split()[0] for l in out.splitlines()), ["AAA.FR", "BBB.FR"])
        self.assertTrue(os.path.exists(target))
        self.assertNotIn("Exporté", out)

    def test_yfinance_logger_silenced_then_restored(self):
        log = logging.getLogger("yfinance")
        before = log.level
        seen = {}

        def fake_scan(*a, **k):
            seen["level"] = log.level
            return pd.DataFrame(), pd.DataFrame(), [], 0
        with mock.patch.object(ds, "run_scan", fake_scan):
            self.run_cli("40", "3m")
        self.assertEqual(seen["level"], logging.CRITICAL)
        self.assertEqual(log.level, before)

    def test_network_warning_survives_quiet_mode(self):
        def fake_scan(*a, **k):
            return pd.DataFrame(), pd.DataFrame(), [f"T{i}.FR" for i in range(60)], 10
        with mock.patch.object(ds, "run_scan", fake_scan):
            _, out, err = self.run_cli("40", "3m")
        self.assertEqual(out, "")
        self.assertIn("ATTENTION", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)