"""
Tests de la conversion XTB -> Yahoo (candidats, résolution, table JSON).
Aucun accès réseau : _download_close est simulé.

    python -m unittest -v test_resolve
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import pandas as pd

try:
    import yfinance  # noqa: F401
except ImportError:
    _stub = types.ModuleType("yfinance")
    _stub.download = lambda *a, **k: pd.DataFrame()
    _stub.Ticker = object
    sys.modules["yfinance"] = _stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402
import scanner as ds  # noqa: E402


def fake_download(alive: set[str], calls: list | None = None):
    def _dl(tickers, start, end):
        if calls is not None:
            calls.append(tuple(tickers))
        cols = {t: [1.0] for t in tickers if t in alive}
        return pd.DataFrame(cols, index=[pd.Timestamp.today().normalize()])
    return _dl


def universe(rows):
    """rows : (ticker XTB, desc) -> DataFrame comme celui de load_universe."""
    df = pd.DataFrame(rows, columns=["ticker", "desc"])
    df["yahoo"] = df["ticker"].map(ds.to_yahoo)
    return df


class TestCandidates(unittest.TestCase):
    def test_first_is_to_yahoo(self):
        for t in ("SAN.FR", "BMW.DE", "NOVOB.DK", "BRKB.US", "AAPL.US"):
            self.assertEqual(ds.candidates(t)[0], ds.to_yahoo(t))

    def test_nordic_class_share(self):
        self.assertEqual(ds.candidates("COLOB.DK"), ["COLOB.CO", "COLO-B.CO"])
        self.assertEqual(ds.candidates("ATCOA.SE"), ["ATCOA.ST", "ATCO-A.ST"])

    def test_us_class_share(self):
        self.assertEqual(ds.candidates("BFB.US"), ["BFB", "BF-B"])

    def test_german_fallback_exchange(self):
        self.assertEqual(ds.candidates("SAP.DE"), ["SAP.DE", "SAP.F"])

    def test_no_dash_variant_for_other_markets(self):
        self.assertEqual(ds.candidates("SAN.FR"), ["SAN.PA"])

    def test_override_is_final(self):
        self.assertEqual(ds.candidates("NOVOB.DK"), ["NOVO-B.CO"])

    def test_unknown(self):
        self.assertEqual(ds.candidates("XYZ.ZZ"), [])
        self.assertEqual(ds.candidates("NODOT"), [])


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch.object(ds.time, "sleep", lambda s: None)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def run_resolve(self, rows, alive, **kw):
        calls = []
        with mock.patch.object(ds, "_download_close", fake_download(alive, calls)):
            return ds.resolve_tickers(universe(rows), **kw), calls

    def test_alive_untouched_dead_corrected(self):
        out, _ = self.run_resolve(
            [("SAN.FR", "Sanofi"), ("COLOB.DK", "Coloplast"), ("GONE.FR", "Delisted")],
            alive={"SAN.PA", "COLO-B.CO"})
        self.assertEqual(out["ok"], 1)
        self.assertEqual(out["resolved"], {"COLOB.DK": "COLO-B.CO"})
        self.assertEqual(list(out["unresolved"]), ["GONE.FR"])
        self.assertEqual(out["unresolved"]["GONE.FR"]["tried"], ["GONE.PA"])

    def test_only_dead_tickers_get_alternatives(self):
        _, calls = self.run_resolve([("SAN.FR", "Sanofi"), ("COLOB.DK", "Coloplast")],
                                    alive={"SAN.PA", "COLO-B.CO"}, retry_rounds=0)
        self.assertEqual(len(calls), 2)            # 1 passe principale + 1 passe alternatives
        self.assertEqual(calls[1], ("COLO-B.CO",))

    def test_no_second_pass_when_all_alive(self):
        _, calls = self.run_resolve([("SAN.FR", "Sanofi")], alive={"SAN.PA"})
        self.assertEqual(len(calls), 1)

    def test_candidate_already_used_by_another_line_is_refused(self):
        out, _ = self.run_resolve([("BFB.US", "Brown-Forman B"), ("BF-B.US", "Autre")],
                                  alive={"BF-B"})
        self.assertNotIn("BFB.US", out["resolved"])

    def test_suggestions_only_when_asked(self):
        rows = [("GONE.FR", "Delisted SA")]
        with mock.patch.object(ds, "_suggest", return_value=["X.PA (Paris, X)"]) as sg:
            out, _ = self.run_resolve(rows, alive=set(), suggest=False)
            sg.assert_not_called()
            out, _ = self.run_resolve(rows, alive=set(), suggest=True)
        self.assertEqual(out["unresolved"]["GONE.FR"]["suggestions"], ["X.PA (Paris, X)"])

    def test_transient_failure_is_retried_not_declared_dead(self):
        seen = {"n": 0}

        def flaky(tickers, start, end):
            seen["n"] += 1
            ok = {"SAN.PA"} if seen["n"] > 1 else set()   # 1er appel : timeout silencieux
            return pd.DataFrame({t: [1.0] for t in tickers if t in ok},
                                index=[pd.Timestamp.today().normalize()])
        with mock.patch.object(ds, "_download_close", flaky):
            out = ds.resolve_tickers(universe([("SAN.FR", "Sanofi")]))
        self.assertEqual(out["ok"], 1)
        self.assertEqual(out["unresolved"], {})

    def test_mass_failure_aborts_instead_of_writing_a_wrong_table(self):
        rows = [(f"T{i}.FR", f"T{i}") for i in range(50)]
        with mock.patch.object(ds, "_download_close",
                               fake_download({f"T{i}.PA" for i in range(20)})):
            with self.assertRaises(RuntimeError):
                ds.resolve_tickers(universe(rows))

    def test_yahoo_down_raises(self):
        def boom(*a, **k):
            raise ConnectionError("down")
        with mock.patch.object(ds, "_download_close", boom):
            with self.assertRaises(RuntimeError):
                ds.resolve_tickers(universe([("SAN.FR", "Sanofi")]))

    def test_empty_big_chunk_raises(self):
        rows = [(f"T{i}.FR", f"T{i}") for i in range(12)]
        with mock.patch.object(ds, "_download_close", fake_download(set())):
            with self.assertRaises(RuntimeError):
                ds.resolve_tickers(universe(rows))


class TestFetchRetry(unittest.TestCase):
    def test_missing_tickers_get_a_second_chance(self):
        state = {"n": 0}
        idx = pd.date_range("2026-10-01", periods=3)

        def fetch(tickers, start, end):
            state["n"] += 1
            bad = {"B"} if state["n"] == 1 else set()   # B échoue au 1er passage
            return pd.DataFrame({t: [1.0, 2.0, 3.0] for t in tickers if t not in bad}, index=idx)
        with mock.patch.object(ds, "fetch_chunk", fetch), \
                mock.patch.object(ds.time, "sleep", lambda s: None):
            out = ds.fetch_prices(["A", "B", "C"], "s", "e")
        self.assertEqual(sorted(out.columns), ["A", "B", "C"])

    def test_dead_ticker_stays_missing_and_retries_are_bounded(self):
        calls = []
        idx = pd.date_range("2026-10-01", periods=3)

        def fetch(tickers, start, end):
            calls.append(tickers)
            return pd.DataFrame({t: [1.0, 2.0, 3.0] for t in tickers if t != "DEAD"}, index=idx)
        with mock.patch.object(ds, "fetch_chunk", fetch), \
                mock.patch.object(ds.time, "sleep", lambda s: None):
            out = ds.fetch_prices(["A", "DEAD"], "s", "e")
        self.assertNotIn("DEAD", out.columns)
        self.assertEqual(len(calls), 1 + ds.RETRY_ROUNDS)

    def test_threads_are_limited(self):
        with mock.patch.object(ds.yf, "download", return_value=pd.DataFrame()) as dl:
            ds._download_close(("A", "B"), "s", "e")
        self.assertEqual(dl.call_args.kwargs["threads"], ds.DOWNLOAD_THREADS)


class TestMapFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "ticker_map.json")

    def read_json(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_roundtrip_and_manual_entries_survive(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"overrides": {"MAN.FR": "MANUAL.PA"}}, f)
        ds.save_ticker_map(self.path, {"COLOB.DK": "COLO-B.CO", "MAN.FR": "AUTO.PA"},
                           {}, checked={"COLOB.DK", "MAN.FR"})
        self.assertEqual(ds.load_ticker_map(self.path),
                         {"COLOB.DK": "COLO-B.CO", "MAN.FR": "MANUAL.PA"})

    def test_unresolved_merge_keeps_unchecked(self):
        ds.save_ticker_map(self.path, {}, {"OLD.FR": {"desc": "o", "tried": [], "suggestions": []}},
                           checked={"OLD.FR"})
        ds.save_ticker_map(self.path, {}, {"NEW.SE": {"desc": "n", "tried": [], "suggestions": []}},
                           checked={"NEW.SE"})
        data = self.read_json()
        self.assertEqual(sorted(data["unresolved"]), ["NEW.SE", "OLD.FR"])

    def test_unresolved_dropped_once_resolved_or_rechecked(self):
        ds.save_ticker_map(self.path, {}, {"A.FR": {"desc": "a", "tried": [], "suggestions": []}},
                           checked={"A.FR"})
        ds.save_ticker_map(self.path, {"A.FR": "A2.PA"}, {}, checked={"A.FR"})
        data = self.read_json()
        self.assertEqual(data["unresolved"], {})
        self.assertEqual(data["overrides"], {"A.FR": "A2.PA"})

    def test_missing_or_corrupt_file(self):
        self.assertEqual(ds.load_ticker_map(self.path), {})
        with open(self.path, "w") as f:
            f.write("{pas du json")
        self.assertEqual(ds.load_ticker_map(self.path), {})

    def test_load_universe_uses_map(self):
        csv = os.path.join(self.dir.name, "u.csv")
        pd.DataFrame({"ticker": ["COLOB.DK", "SAN.FR"], "desc": ["Coloplast", "Sanofi"],
                      "section": ["Stock", "Stock"], "currency": ["DKK", "EUR"]}
                     ).to_csv(csv, index=False)
        plain = ds.load_universe(csv).set_index("ticker")["yahoo"]
        mapped = ds.load_universe(csv, {"COLOB.DK": "COLO-B.CO"}).set_index("ticker")["yahoo"]
        self.assertEqual(plain["COLOB.DK"], "COLOB.CO")
        self.assertEqual(mapped["COLOB.DK"], "COLO-B.CO")
        self.assertEqual(mapped["SAN.FR"], "SAN.PA")


class TestCliResolve(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.csv = os.path.join(self.dir.name, "u.csv")
        pd.DataFrame({"ticker": ["SAN.FR", "COLOB.DK", "GONE.FR"],
                      "desc": ["Sanofi", "Coloplast", "Delisted"],
                      "section": ["Stock"] * 3, "currency": ["EUR", "DKK", "EUR"]}
                     ).to_csv(self.csv, index=False)

    def run_cli(self, *args, alive=()):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["drawdown_cli.py", *args]), \
                mock.patch.object(ds, "_download_close", fake_download(set(alive))), \
                mock.patch.object(ds.time, "sleep", lambda s: None), \
                redirect_stdout(out), redirect_stderr(err):
            code = cli.main()
        return code, out.getvalue(), err.getvalue()

    def test_resolve_writes_map_next_to_csv_and_does_not_scan(self):
        code, out, _ = self.run_cli("--resolve", "--csv", self.csv,
                                    alive={"SAN.PA", "COLO-B.CO"})
        self.assertEqual(code, 0)
        self.assertIn("COLOB.DK", out)
        self.assertIn("COLO-B.CO", out)
        self.assertIn("GONE.FR", out)
        mp = os.path.join(self.dir.name, ds.MAP_FILE)
        self.assertEqual(ds.load_ticker_map(mp), {"COLOB.DK": "COLO-B.CO"})

    def test_resolve_needs_no_drawdown_arguments(self):
        with mock.patch("builtins.input", side_effect=AssertionError("ne doit rien demander")):
            code, _, _ = self.run_cli("--resolve", "--csv", self.csv, alive={"SAN.PA"})
        self.assertEqual(code, 0)

    def test_second_run_uses_saved_map(self):
        self.run_cli("--resolve", "--csv", self.csv, alive={"SAN.PA", "COLO-B.CO"})
        _, out, _ = self.run_cli("--resolve", "--csv", self.csv, alive={"SAN.PA", "COLO-B.CO"})
        self.assertIn("2 OK", out)  # COLOB.DK est maintenant converti d'office en COLO-B.CO


if __name__ == "__main__":
    unittest.main(verbosity=2)