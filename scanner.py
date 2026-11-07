"""
Scanner de drawdown sur l'univers XTB (CSV) avec yfinance, interface Streamlit.

Installation :  pip install streamlit yfinance pandas numpy
Lancement    :  streamlit run drawdown_scanner.py

Place le CSV XTB à côté du script (nom par défaut ci-dessous) ou charge-le
depuis la barre latérale. Les CFD sont toujours exclus.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from cache import DiskCache

st = None

DEFAULT_CSV = "xtb_stocks_etfs.csv"
CACHE_DIR = ".cache"

# Cache disque (voir cache.py). Désactivé par défaut : le CLI l'active via configure_cache().
CACHE: DiskCache | None = None


def configure_cache(root, **options) -> DiskCache | None:
    """Active le cache disque dans `root` (None pour le désactiver)."""
    global CACHE
    CACHE = DiskCache(root, **options) if root else None
    return CACHE


# Téléchargement : peu de requêtes en parallèle (trop de threads => timeouts DNS / limites Yahoo
# et des titres existants sont alors pris à tort pour « delisted »), puis seconde chance.
DOWNLOAD_THREADS = 4
RETRY_ROUNDS = 2   # passes de rattrapage sur les titres sans cours
RETRY_CHUNK = 10   # petits lots pendant le rattrapage

# Suffixe de marché XTB -> suffixe Yahoo Finance
YAHOO_SUFFIX = {
    "US": "", "FR": ".PA", "DE": ".DE", "ES": ".MC", "IT": ".MI", "NL": ".AS",
    "BE": ".BR", "PT": ".LS", "FI": ".HE", "SE": ".ST", "DK": ".CO",
    "NO": ".OL", "CH": ".SW", "UK": ".L", "CZ": ".PR",
}
# Marchés de l'EEE (approximation "PEA probable", voir avertissement dans l'UI)
EEA_MARKETS = {"FR", "DE", "ES", "IT", "NL", "BE", "PT", "FI", "SE", "DK", "NO"}

# ETF à effet de levier / inverses, repérés par leur nom (heuristique).
# "Ultra Short-Term Bond" ou "Short Term Treasury" ne sont PAS à levier : ils sont épargnés.
LEVERAGED_RE = re.compile(
    r"\b[1-5]x\b|\bbull\b|\bbear\b|\binverse\b|\bleverag|\blevered\b"
    r"|\bultra(?!\s*short[- ]?term)|proshares\s+short\b",
    re.IGNORECASE,
)

# Corrections manuelles quand la règle générale ne suffit pas (actions de classe B, etc.)
# Complète cette liste avec les tickers listés dans "Sans données" après un scan.
OVERRIDES = {
    "BRKB.US": "BRK-B",
    "ERICB.SE": "ERIC-B.ST",
    "NOVOB.DK": "NOVO-B.CO",
    "VOLVB.SE": "VOLV-B.ST",
}


# --------------------------------------------------------------------------- #
# Univers
# --------------------------------------------------------------------------- #
def to_yahoo(xtb_ticker: str) -> str | None:
    """Convertit un ticker XTB (ex. 'SAN.FR') en ticker Yahoo (ex. 'SAN.PA')."""
    if xtb_ticker in OVERRIDES:
        return OVERRIDES[xtb_ticker]
    if "." not in xtb_ticker:
        return None
    base, market = xtb_ticker.rsplit(".", 1)
    if market not in YAHOO_SUFFIX:
        return None
    return base.replace(".", "-") + YAHOO_SUFFIX[market]


# --------------------------------------------------------------------------- #
# Conversion XTB -> Yahoo : candidats, validation, table de correspondance
# --------------------------------------------------------------------------- #
MAP_FILE = "ticker_map.json"
# Classes d'actions écrites avec un tiret chez Yahoo (NOVO-B.CO, ERIC-B.ST, BRK-B)
DASH_CLASS_LETTERS = {"US": "ABC", "DK": "ABCDKRV", "SE": "ABCDKRV",
                      "NO": "ABCDKRV", "FI": "ABCDKRV"}
ALT_SUFFIX = {"DE": [".F"]}  # place de repli quand Xetra n'a rien


def candidates(xtb_ticker: str) -> list[str]:
    """Tickers Yahoo possibles pour un ticker XTB, du plus au moins probable.
    Le premier est celui de to_yahoo(). Les suivants ne servent que si lui-même
    ne renvoie aucun cours (voir resolve_tickers)."""
    first = to_yahoo(xtb_ticker)
    if first is None:
        return []
    if xtb_ticker in OVERRIDES:
        return [first]
    base, market = xtb_ticker.rsplit(".", 1)
    base = base.replace(".", "-")
    suffix = YAHOO_SUFFIX[market]
    out = [first]
    letters = DASH_CLASS_LETTERS.get(market, "")
    if letters and len(base) >= 3 and "-" not in base and base[-1] in letters:
        out.append(f"{base[:-1]}-{base[-1]}{suffix}")  # NOVOB -> NOVO-B
    out += [base + alt for alt in ALT_SUFFIX.get(market, [])]
    return list(dict.fromkeys(out))


def load_ticker_map(path) -> dict[str, str]:
    """Corrections XTB -> Yahoo enregistrées dans le fichier JSON (modifiables à la main)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    overrides = data.get("overrides", {}) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in overrides.items() if v}


def save_ticker_map(path, resolved: dict, unresolved: dict, checked) -> None:
    """Fusionne les résultats d'une résolution dans le JSON. Les corrections déjà
    présentes (donc éventuellement faites à la main) ne sont jamais écrasées ; la liste
    « unresolved » garde les titres non revérifiés cette fois-ci."""
    p = Path(path)
    overrides = load_ticker_map(p)
    overrides.update({k: v for k, v in resolved.items() if k not in overrides})
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8")).get("unresolved", {})
        except (OSError, ValueError, AttributeError):
            prev = {}
    keep = {k: v for k, v in prev.items() if k not in set(checked) and k not in overrides}
    keep.update(unresolved)
    data = {"updated": date.today().isoformat(),
            "overrides": dict(sorted(overrides.items())),
            "unresolved": dict(sorted(keep.items()))}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _alive(tickers: list[str], days: int, chunk: int,
           retry_rounds: int = RETRY_ROUNDS, guard: bool = True) -> set[str]:
    """Tickers Yahoo ayant au moins un cours sur les `days` derniers jours.
    Les titres sans cours sont retentés en petits lots avant d'être déclarés morts.
    Lève RuntimeError si Yahoo ne répond pas (ou, avec guard, si plus d'un quart des titres
    n'ont pas de cours : problème réseau probable), pour ne pas enregistrer une table fausse."""
    end = pd.Timestamp.today().normalize()
    start_s = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end_s = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    alive: set[str] = set()
    for i in range(0, len(tickers), chunk):
        part = tuple(tickers[i:i + chunk])
        got = None
        for _ in range(3):
            try:
                got = {str(c) for c in _download_close(part, start_s, end_s).columns}
            except Exception:
                got = None
            if got:
                break
            time.sleep(2)
        if got is None:
            raise RuntimeError("Yahoo injoignable : abandon pour ne pas enregistrer une table fausse.")
        if not got and len(part) >= 10:
            raise RuntimeError("Yahoo n'a renvoyé aucun cours pour tout un lot "
                               "(limite de débit ?). Réessaie plus tard.")
        alive |= got

    dead = [t for t in tickers if t not in alive]
    for rnd in range(retry_rounds):
        if not dead:
            break
        time.sleep(2 * (rnd + 1))
        for i in range(0, len(dead), RETRY_CHUNK):
            try:
                alive |= {str(c) for c in
                          _download_close(tuple(dead[i:i + RETRY_CHUNK]), start_s, end_s).columns}
            except Exception:
                pass
        dead = [t for t in tickers if t not in alive]
    if guard and len(tickers) >= 40 and len(dead) > 0.25 * len(tickers):
        raise RuntimeError(f"{len(dead)}/{len(tickers)} tickers sans cours : probablement un "
                           "problème réseau (DNS ?), pas des titres disparus. Abandon pour ne "
                           "pas enregistrer une table fausse.")
    return alive


def _suggest(desc: str, limit: int = 3) -> list[str]:
    """Pistes de tickers Yahoo à partir du nom (à vérifier à la main, jamais appliquées seules)."""
    search = getattr(yf, "Search", None)
    if search is None or not desc:
        return []
    try:
        quotes = search(desc, max_results=limit).quotes
    except Exception:
        return []
    return [f"{q['symbol']} ({q.get('exchDisp') or q.get('exchange') or '?'}, "
            f"{q.get('shortname') or q.get('longname') or '?'})"
            for q in quotes if q.get("symbol")]


def resolve_tickers(universe: pd.DataFrame, days: int = 21, chunk: int = 100,
                    suggest: bool = False, retry_rounds: int = RETRY_ROUNDS) -> dict:
    """Vérifie chaque ticker Yahoo de l'univers (colonnes ticker, desc, yahoo) puis, pour
    ceux sans cours récent, essaie les candidats alternatifs de candidates().
    Retourne {"checked", "ok", "resolved": {xtb: yahoo}, "unresolved": {xtb: {...}}}."""
    alive = _alive(universe["yahoo"].tolist(), days, chunk, retry_rounds)
    dead = universe[~universe["yahoo"].isin(alive)]
    alternatives = {xtb: [c for c in candidates(xtb) if c != y]
                    for xtb, y in zip(dead["ticker"], dead["yahoo"])}
    pool = sorted({c for alts in alternatives.values() for c in alts})
    # alternatives : la plupart sont mortes par construction, donc ni garde-fou ni insistance
    alive_alt = _alive(pool, days, chunk, min(retry_rounds, 1), guard=False) if pool else set()
    taken = set(universe["yahoo"])  # un ticker Yahoo ne doit servir qu'à un seul titre XTB
    resolved, unresolved = {}, {}
    for xtb, y, desc in zip(dead["ticker"], dead["yahoo"], dead["desc"]):
        hit = next((c for c in alternatives[xtb] if c in alive_alt and c not in taken), None)
        if hit:
            resolved[xtb] = hit
            taken.add(hit)
        else:
            unresolved[xtb] = {"desc": desc, "tried": [y, *alternatives[xtb]],
                               "suggestions": _suggest(desc) if suggest else []}
    return {"checked": set(universe["ticker"]), "ok": len(alive & set(universe["yahoo"])),
            "resolved": resolved, "unresolved": unresolved}


def load_universe(src, ticker_map: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(src)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["desc"] = df["desc"].fillna("").astype(str)
    df["close_only"] = df["ticker"].str.endswith("*")  # titres en "CLOSE ONLY"
    df["ticker"] = df["ticker"].str.rstrip("*")
    df["market"] = df["ticker"].str.rsplit(".", n=1).str[-1]
    df = df[~df["desc"].str.contains("CFD", case=False)].copy()  # CFD : toujours exclus
    df["is_leveraged"] = (df["section"] == "ETF") & df["desc"].str.contains(LEVERAGED_RE)
    tmap = ticker_map or {}
    df["yahoo"] = df["ticker"].map(lambda t: tmap.get(t) or to_yahoo(t))
    df["pea_probable"] = df["market"].isin(EEA_MARKETS) & (
        (df["section"] != "ETF") | df["desc"].str.upper().str.contains("PEA")
    )
    df = df.dropna(subset=["yahoo"]).sort_values("close_only")
    df = df.drop_duplicates("ticker").drop_duplicates("yahoo")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #
def _cache(func):
    return st.cache_data(show_spinner=False, ttl=3600)(func) if st else func


def _download_close(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers), start=start, end=end,
        auto_adjust=True, progress=False, threads=DOWNLOAD_THREADS,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"]
    if isinstance(close, pd.Series):  # un seul ticker, anciennes versions
        close = close.to_frame(tickers[0])
    return close.dropna(axis=1, how="all")


fetch_chunk = _cache(_download_close)


def fetch_prices(tickers, start, end, progress_cb=None, chunk=100,
                 retry_rounds=RETRY_ROUNDS) -> pd.DataFrame:
    """Cours de clôture ajustés. Passe par le cache disque s'il est activé."""
    if CACHE is None:
        return _fetch_network(tickers, start, end, progress_cb, chunk, retry_rounds)

    def network(part, g_start, g_end, cb):
        return _fetch_network(part, g_start, g_end, cb, chunk, retry_rounds)
    return CACHE.fetch_prices(list(tickers), start, end, network, progress_cb)


def _fetch_network(tickers, start, end, progress_cb=None, chunk=100,
                   retry_rounds=RETRY_ROUNDS) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tuple(tickers[i:i + chunk])
        for _ in range(2):  # un essai de plus en cas d'erreur réseau / limite Yahoo
            try:
                frames.append(fetch_chunk(part, start, end))
                break
            except Exception:
                time.sleep(2)
        if progress_cb:
            progress_cb(min(i + chunk, len(tickers)) / len(tickers))
    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Seconde chance : yfinance ne lève pas d'erreur quand une requête échoue (timeout DNS...),
    # le titre manque simplement. On retente ceux-là en petits lots ; un vrai titre mort
    # échoue vite et reste manquant.
    missing = [t for t in tickers if t not in prices.columns]
    for rnd in range(retry_rounds):
        if not missing:
            break
        time.sleep(2 * (rnd + 1))
        extra = []
        for i in range(0, len(missing), RETRY_CHUNK):
            try:
                extra.append(fetch_chunk(tuple(missing[i:i + RETRY_CHUNK]), start, end))
            except Exception:
                pass
        extra = [f for f in extra if not f.empty]
        if extra:
            prices = pd.concat([prices, *extra], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()]
        missing = [t for t in tickers if t not in prices.columns]

    if prices.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(prices.index)
    prices.index = idx.tz_localize(None) if idx.tz is not None else idx
    return prices.sort_index()


# --------------------------------------------------------------------------- #
# PER (Price/Earnings)
# --------------------------------------------------------------------------- #
_PE_CACHE: dict[str, tuple[float, float]] = {}
_PE_NEW: dict[str, tuple[float, float]] = {}  # reçus du réseau, à écrire sur disque


def _to_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_pe(yahoo_ticker: str) -> tuple[float, float]:
    """(PER sur 12 derniers mois, PER prévu). NaN si Yahoo n'a pas la donnée
    (bénéfices négatifs, ETF, titre mal couvert...)."""
    if yahoo_ticker not in _PE_CACHE:
        hit = CACHE.get_pe(yahoo_ticker) if CACHE else None
        if hit is not None:
            _PE_CACHE[yahoo_ticker] = hit
        else:
            try:
                info = yf.Ticker(yahoo_ticker).info
                if not info:
                    raise ValueError("réponse vide")
                _PE_CACHE[yahoo_ticker] = (_to_float(info.get("trailingPE")),
                                           _to_float(info.get("forwardPE")))
                _PE_NEW[yahoo_ticker] = _PE_CACHE[yahoo_ticker]
            except Exception:  # échec : retenu pour la session seulement, jamais sur disque
                _PE_CACHE[yahoo_ticker] = (np.nan, np.nan)
    return _PE_CACHE[yahoo_ticker]


def add_pe(res: pd.DataFrame, progress_cb=None, workers: int = 8) -> pd.DataFrame:
    """Ajoute les colonnes PER / PER prévu (une requête par titre, en parallèle).
    Appelé uniquement sur la short-list après filtre drawdown, pour rester rapide.
    Le PER n'a de sens que pour les actions : les ETF ne sont pas interrogés (NaN)."""
    tickers = res.loc[res["section"] == "Stock", "yahoo"].tolist()
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(get_pe, y): y for y in tickers}
        for i, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if progress_cb:
                progress_cb(i / len(tickers))
    if CACHE and _PE_NEW:
        CACHE.save_pe(dict(_PE_NEW))
        _PE_NEW.clear()
    res = res.copy()
    res["PER"] = res["yahoo"].map(lambda y: out.get(y, (np.nan, np.nan))[0])
    res["PER prévu"] = res["yahoo"].map(lambda y: out.get(y, (np.nan, np.nan))[1])
    return res


def filter_pe(res: pd.DataFrame, pe_min: float, pe_max: float,
              pe_col: str, keep_missing: bool) -> pd.DataFrame:
    """Filtre sur le PER. S'applique aux actions uniquement : les ETF passent toujours."""
    pe = res[pe_col]
    ok = pe.between(pe_min, pe_max)  # NaN -> False
    if keep_missing:
        ok = ok | pe.isna()
    ok = ok | (res["section"] != "Stock")
    return res[ok]


# --------------------------------------------------------------------------- #
# Calculs
# --------------------------------------------------------------------------- #
def compute_metrics(close: pd.Series, window_start: pd.Timestamp,
                    min_points: int, last_day: pd.Timestamp) -> dict | None:
    s = close.dropna()
    if s.empty or s.index[-1] < last_day - pd.Timedelta(days=7):
        return None  # cotation arrêtée ou données obsolètes
    w = s[s.index >= window_start]
    if len(w) < max(min_points, 5):
        return None  # historique trop court sur la fenêtre
    dd = w / w.cummax() - 1.0
    last, high = w.iloc[-1], w.max()
    ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
    return {
        "Dernier cours": last,
        "Plus haut période": high,
        "Date plus haut": w.idxmax().date(),
        "DD actuel %": (last / high - 1) * 100,
        "DD max %": dd.min() * 100,
        "Date creux": dd.idxmin().date(),
        "Perf période %": (last / w.iloc[0] - 1) * 100,
        "vs MM200 %": (last / ma200 - 1) * 100 if pd.notna(ma200) else np.nan,
    }


def run_scan(universe: pd.DataFrame, window_start: pd.Timestamp, end: pd.Timestamp,
             dd_threshold: float, dd_column: str, above_ma200: bool,
             progress_cb=None, pe_max: float | None = None, pe_min: float = 0.0,
             pe_col: str = "PER", keep_missing_pe: bool = False,
             show_pe: bool = False, pe_progress_cb=None):
    """Retourne (résultats filtrés, prix, liste des tickers sans données, nb analysés)."""
    # On charge au moins ~14 mois pour pouvoir calculer la MM200
    fetch_start = min(window_start, end - pd.Timedelta(days=420))
    tickers = universe["yahoo"].tolist()
    prices = fetch_prices(
        tickers, fetch_start.strftime("%Y-%m-%d"),
        (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress_cb,
    )
    if prices.empty:
        return pd.DataFrame(), prices, universe["ticker"].tolist(), 0

    expected = (end - window_start).days * 252 / 365
    min_points = int(0.7 * expected)
    last_day = prices.index.max()

    rows, missing = [], []
    for y in tickers:
        m = compute_metrics(prices[y], window_start, min_points, last_day) \
            if y in prices.columns else None
        if m is None:
            missing.append(y)
        else:
            rows.append({"yahoo": y, **m})

    info = universe[["yahoo", "ticker", "desc", "section", "currency", "market"]]
    missing_xtb = info.loc[info["yahoo"].isin(missing), "ticker"].tolist()
    if not rows:
        return pd.DataFrame(), prices, missing_xtb, 0

    res = pd.DataFrame(rows).merge(info, on="yahoo")
    analysed = len(res)
    res = res[res[dd_column] <= -dd_threshold]
    if above_ma200:
        res = res[res["vs MM200 %"] > 0]
    res = res.sort_values(dd_column).reset_index(drop=True)
    # PER : récupéré seulement pour les titres déjà retenus (une requête par titre)
    if not res.empty and (pe_max is not None or show_pe):
        res = add_pe(res, pe_progress_cb)
        if pe_max is not None:
            res = filter_pe(res, pe_min, pe_max, pe_col, keep_missing_pe)
            res = res.reset_index(drop=True)
    return res, prices, missing_xtb, analysed


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Scanner drawdown XTB", layout="wide")
    st.title("Scanner de drawdown — univers XTB")

    with st.sidebar:
        st.header("Paramètres")
        up = st.file_uploader("CSV XTB (optionnel)", type="csv")
        dd_threshold = st.slider("Drawdown minimum X (%)", 5, 80, 20)
        c1, c2 = st.columns(2)
        n = c1.number_input("Durée Y", min_value=1, max_value=120, value=3)
        unit = c2.selectbox("Unité", ["jours", "mois", "ans"], index=1)
        mode = st.radio(
            "Type de drawdown",
            ["Actuel (cours vs plus haut de la période)",
             "Maximal (pire chute pic → creux sur la période)"],
        )
        types = st.multiselect("Type d'actif", ["Stock", "ETF"], default=["Stock"])
        excl_lev = st.checkbox("Exclure les ETF à effet de levier / inverses", True)
        pea_only = st.checkbox("PEA probable uniquement (approximation)", False)
        above_ma = st.checkbox("Garder seulement ceux au-dessus de la MM200", False)
        use_pe = st.checkbox("Filtrer sur le PER (actions uniquement, les ETF ne sont pas filtrés)", False)
        pe_max, pe_col, keep_missing = None, "PER", False
        if use_pe:
            pe_max = st.slider("PER maximum", 5, 60, 20)
            pe_col = "PER" if st.radio(
                "Type de PER", ["Actuel (12 derniers mois)", "Prévu (forward)"]
            ).startswith("Actuel") else "PER prévu"
            keep_missing = st.checkbox("Garder les titres sans PER disponible", False)
        limit = st.number_input("Limiter à N titres (0 = tous, utile pour tester)",
                                min_value=0, value=0, step=50)

    src = up if up is not None else DEFAULT_CSV
    try:
        uni = load_universe(src, load_ticker_map(MAP_FILE))
    except FileNotFoundError:
        st.error(f"Fichier '{DEFAULT_CSV}' introuvable : charge le CSV dans la barre latérale.")
        return

    markets = sorted(uni["market"].unique())
    with st.sidebar:
        sel_markets = st.multiselect("Marchés", markets, default=markets)

    uni = uni[uni["section"].isin(types) & uni["market"].isin(sel_markets)
              & ~uni["close_only"]]
    if excl_lev:
        uni = uni[~uni["is_leveraged"]]
    if pea_only:
        uni = uni[uni["pea_probable"]]
    if limit:
        uni = uni.head(int(limit))
    uni = uni.reset_index(drop=True)

    st.caption(f"{len(uni)} titres seront analysés (≈ {max(1, len(uni) // 60)} min). "
               "Les CFD sont toujours exclus.")

    end = pd.Timestamp.today().normalize()
    offset = {"jours": pd.Timedelta(days=int(n)),
              "mois": pd.DateOffset(months=int(n)),
              "ans": pd.DateOffset(years=int(n))}[unit]
    window_start = end - offset
    dd_col = "DD actuel %" if mode.startswith("Actuel") else "DD max %"

    if st.button("🔍 Lancer le scan", type="primary", disabled=uni.empty):
        bar = st.progress(0.0, text="Téléchargement des cours…")
        res, prices, missing, analysed = run_scan(
            uni, window_start, end, dd_threshold, dd_col, above_ma,
            progress_cb=lambda p: bar.progress(p, text="Téléchargement des cours…"),
            pe_max=pe_max, pe_col=pe_col, keep_missing_pe=keep_missing,
            pe_progress_cb=lambda p: bar.progress(p, text="Récupération des PER…"),
        )
        bar.empty()
        st.session_state["scan"] = {
            "res": res, "missing": missing, "analysed": analysed, "col": dd_col,
            "prices": prices[res["yahoo"]] if not res.empty else pd.DataFrame(),
            "window_start": window_start,
        }

    scan = st.session_state.get("scan")
    if not scan:
        return

    res = scan["res"]
    st.subheader(f"{len(res)} titres sous -{dd_threshold} % "
                 f"({scan['analysed']} analysés, {len(scan['missing'])} sans données)")

    if not res.empty:
        cols = ["ticker", "desc", "section", "currency", "market", scan["col"],
                "DD actuel %", "DD max %", "Perf période %", "vs MM200 %"]
        cols += [c for c in ("PER", "PER prévu") if c in res.columns]
        cols += ["Plus haut période", "Date plus haut", "Date creux"]
        cols = list(dict.fromkeys(cols))  # dédoublonne en gardant l'ordre
        table = res[cols].rename(columns={
            "ticker": "Ticker XTB", "desc": "Nom", "section": "Type",
            "currency": "Devise", "market": "Marché"})
        num = table.select_dtypes("number").columns
        table[num] = table[num].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Exporter en CSV", table.to_csv(index=False).encode(),
                           "drawdown_scan.csv", "text/csv")

        choice = st.selectbox("Voir le graphique de :", res["ticker"] + " — " + res["desc"])
        y = res.loc[(res["ticker"] + " — " + res["desc"]) == choice, "yahoo"].iloc[0]
        series = scan["prices"][y].dropna()
        st.line_chart(series[series.index >= scan["window_start"]])

    with st.expander("Titres sans données (à corriger dans OVERRIDES si besoin)"):
        st.write(", ".join(scan["missing"]) or "Aucun")

    st.info(
        "Points d'attention : prix ajustés des dividendes, exprimés en devise locale "
        "(le change n'est pas pris en compte). « PEA probable » se base sur la place de "
        "cotation et le mot PEA dans le nom des ETF : vérifie l'éligibilité sur la fiche "
        "du titre. Yahoo ne couvre pas tous les tickers XTB. Le PER vient de Yahoo : absent "
        "si les bénéfices sont négatifs, et peu comparable d'un secteur à l'autre."
    )


if __name__ == "__main__":
    if st is None:
        print("Streamlit n'est pas installé. Pour la version CLI : python drawdown_cli.py -h")
    else:
        main()