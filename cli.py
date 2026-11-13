"""
Version ligne de commande du scanner de drawdown XTB (yfinance).

Installation :  pip install yfinance pandas numpy
Exemples :
    python drawdown_cli.py 20 3m                 # drawdown >= 20 % sur 3 mois
    python drawdown_cli.py 30 1y --mode max --pea --above-ma200
    python drawdown_cli.py 25 6m --pe-max 15     # + filtre PER <= 15 (sur la short-list)
    python drawdown_cli.py 25 6m --roic-min 10   # + filtre ROIC moyen >= 10 % (sur la short-list)
    python drawdown_cli.py                       # mode interactif (questions)
    python drawdown_cli.py --resolve             # vérifie/corrige la conversion XTB -> Yahoo
    python drawdown_cli.py 30 6m --cache-ttl 72  # cours réutilisés jusqu'à 72 h sans requête
    python drawdown_cli.py 30 6m --refresh       # ignore le cache et retélécharge tout
    python drawdown_cli.py --resolve --markets SE DK FI --suggest

Par défaut la sortie est la seule liste des titres retenus (une ligne « ticker  nom », le plus
touché d'abord) ; -v / --verbose rétablit la progression, le bilan du cache et le tableau détaillé.
Les CFD sont toujours exclus.
Durée : 90d (jours), 12w (semaines), 3m (mois), 1y (ans). "j" et "a" acceptés.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import scanner as ds

UNITS = {"d": "days", "j": "days", "w": "weeks", "m": "months", "y": "years", "a": "years"}


def parse_duration(text: str) -> pd.DateOffset | pd.Timedelta:
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]+)\s*", text)
    if not m or m.group(2).lower() not in UNITS:
        raise argparse.ArgumentTypeError("durée invalide, ex. : 90d, 12w, 3m, 1y")
    n, unit = int(m.group(1)), UNITS[m.group(2).lower()]
    if unit == "days":
        return pd.Timedelta(days=n)
    if unit == "weeks":
        return pd.Timedelta(weeks=n)
    return pd.DateOffset(**{unit: n})


# --- Export du projet pour un LLM -------------------------------------------
DUMP_NAME = "forllm.txt"
DUMP_EXTS = {".py", ".md", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml", ".sh", ".txt"}
DUMP_SKIP_DIRS = {"__pycache__", "venv", ".venv", "env", "node_modules", "build", "dist"}
DUMP_MAX_BYTES = 500_000  # ignore les gros fichiers (données, exports)
DUMP_SKIP_NAMES = {ds.MAP_FILE}  # table de conversion : donnée, pas du code


def dump_for_llm(root: Path | None = None, name: str = DUMP_NAME) -> Path:
    """Concatène tous les fichiers texte du projet dans <root>/forllm.txt."""
    root = (root or Path(__file__).resolve().parent).resolve()
    target = root / name
    files = []
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if not f.is_file() or f == target:
            continue
        if any(part.startswith(".") or part in DUMP_SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if f.name.startswith(".") or f.name in DUMP_SKIP_NAMES or f.suffix.lower() not in DUMP_EXTS:
            continue
        if f.stat().st_size > DUMP_MAX_BYTES:
            continue
        files.append(f)

    parts = [f"# Projet : {root.name}",
             f"# Généré le {datetime.now():%Y-%m-%d %H:%M}",
             "# Fichiers : " + ", ".join(f.relative_to(root).as_posix() for f in files),
             ""]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binaire ou illisible
        parts.append(f"===== {f.relative_to(root).as_posix()} =====")
        parts.append(text.rstrip("\n"))
        parts.append("")
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scanner de drawdown sur l'univers XTB.")
    p.add_argument("drawdown", nargs="?", type=float, help="drawdown minimum en %% (ex. 20)")
    p.add_argument("duree", nargs="?", help="durée de la fenêtre (ex. 3m, 90d, 1y)")
    p.add_argument("--csv", default=ds.DEFAULT_CSV, help="chemin du CSV XTB")
    p.add_argument("--map", help=f"table de correspondance XTB -> Yahoo "
                                 f"(défaut : {ds.MAP_FILE} à côté du CSV)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="affiche progression, bilan du cache et tableau détaillé "
                        "(par défaut : seulement ticker et nom)")
    p.add_argument("--no-cache", action="store_true", help="désactive le cache disque")
    p.add_argument("--refresh", action="store_true",
                   help="ignore le cache en lecture et retélécharge tout (le cache est réécrit)")
    p.add_argument("--cache-dir", help=f"dossier du cache (défaut : {ds.CACHE_DIR} à côté du CSV)")
    p.add_argument("--cache-ttl", type=float, default=12.0, metavar="HEURES",
                   help="durée pendant laquelle les cours en cache sont réutilisés sans "
                        "aucune requête (défaut 12 ; au-delà, seuls les derniers jours "
                        "sont retéléchargés)")
    p.add_argument("--resolve", action="store_true",
                   help="vérifie les tickers Yahoo de l'univers filtré, corrige ceux qui "
                        "n'ont pas de cours (ex. NOVOB.DK -> NOVO-B.CO), enregistre la table, "
                        "puis s'arrête (pas de scan)")
    p.add_argument("--suggest", action="store_true",
                   help="avec --resolve : propose des tickers via la recherche Yahoo "
                        "pour les titres non résolus (plus lent)")
    p.add_argument("--mode", choices=["actuel", "max"], default="actuel",
                   help="actuel = cours vs plus haut de la période ; max = pire chute pic→creux")
    p.add_argument("--types", nargs="+", choices=["Stock", "ETF"], default=["Stock"])
    p.add_argument("--markets", nargs="+", help="ex. FR DE US (défaut : tous)")
    p.add_argument("--with-leveraged", action="store_true",
                   help="inclure les ETF à levier / inverses (exclus par défaut)")
    p.add_argument("--pea", action="store_true", help="PEA probable uniquement (approximation)")
    p.add_argument("--above-ma200", action="store_true", help="garder les titres au-dessus de la MM200")
    p.add_argument("--pe-max", type=float, help="PER maximum (ex. 20), actions uniquement : les ETF ne sont pas filtrés")
    p.add_argument("--pe-min", type=float, default=0.0, help="PER minimum (défaut 0)")
    p.add_argument("--pe-type", choices=["trailing", "forward"], default="trailing",
                   help="PER sur 12 mois (trailing) ou prévu (forward)")
    p.add_argument("--keep-no-pe", action="store_true", help="garder les titres sans PER")
    p.add_argument("--show-pe", action="store_true", help="afficher le PER sans filtrer")
    p.add_argument("--roic-min", type=float,
                   help="ROIC minimum en %% (ex. 10), actions uniquement : les ETF ne sont pas filtrés. "
                        "Calculé depuis les comptes annuels Yahoo, sur la short-list seulement")
    p.add_argument("--roic-type", choices=["avg", "last"], default="avg",
                   help="ROIC moyen des derniers exercices (avg, défaut) ou du dernier exercice (last)")
    p.add_argument("--keep-no-roic", action="store_true", help="garder les titres sans ROIC")
    p.add_argument("--show-roic", action="store_true", help="afficher le ROIC sans filtrer")
    p.add_argument("--limit", type=int, default=0, help="limiter à N titres (test rapide)")
    p.add_argument("--out", help="exporter le résultat en CSV")
    return p


def run_resolve(uni: pd.DataFrame, map_path: str, suggest: bool) -> int:
    print(f"{len(uni)} tickers à vérifier (cours des 21 derniers jours)…", file=sys.stderr)
    try:
        out = ds.resolve_tickers(uni, suggest=suggest)
    except RuntimeError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    ds.save_ticker_map(map_path, out["resolved"], out["unresolved"], out["checked"])
    resolved, unresolved = out["resolved"], out["unresolved"]
    print(f"{len(uni)} titres vérifiés : {out['ok']} OK, {len(resolved)} corrigés "
          f"automatiquement, {len(unresolved)} sans correspondance")
    if resolved:
        print("\nCorrigés :")
        for xtb, y in sorted(resolved.items()):
            print(f"  {xtb:<14} -> {y}")
    if unresolved:
        print("\nSans correspondance (corrige à la main dans \"overrides\" du JSON) :")
        for xtb, info in sorted(unresolved.items()):
            print(f"  {xtb:<14} {info['desc'][:40]}")
            for sug in info["suggestions"]:
                print(f"{'':18}? {sug}")
    print(f"\nTable enregistrée : {map_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if not args.no_cache and not args.resolve:
        root = args.cache_dir or Path(args.csv).resolve().parent / ds.CACHE_DIR
        ds.configure_cache(root, ttl_hours=args.cache_ttl, refresh=args.refresh)
    yf_log = logging.getLogger("yfinance")
    previous_level = yf_log.level
    if not args.verbose:
        yf_log.setLevel(logging.CRITICAL)  # plus de « possibly delisted » ni d'erreurs curl
    try:
        return _main(args)
    finally:
        yf_log.setLevel(previous_level)
        ds.configure_cache(None)  # pas d'état global qui traîne entre deux appels


def _main(args) -> int:
    if args.drawdown is None and not args.resolve:
        args.drawdown = float(input("Drawdown minimum (%) : ").replace(",", "."))
    if args.duree is None and not args.resolve:
        args.duree = input("Durée (ex. 3m, 90d, 1y) : ")
    map_path = args.map or str(Path(args.csv).with_name(ds.MAP_FILE))

    try:
        uni = ds.load_universe(args.csv, ds.load_ticker_map(map_path))
    except FileNotFoundError:
        sys.exit(f"CSV introuvable : {args.csv} (option --csv)")
    uni = uni[uni["section"].isin(args.types) & ~uni["close_only"]]
    if args.markets:
        uni = uni[uni["market"].isin([m.upper() for m in args.markets])]
    if not args.with_leveraged:
        uni = uni[~uni["is_leveraged"]]
    if args.pea:
        uni = uni[uni["pea_probable"]]
    if args.limit:
        uni = uni.head(args.limit)
    uni = uni.reset_index(drop=True)
    if uni.empty:
        sys.exit("Aucun titre après filtres.")
    if args.resolve:
        return run_resolve(uni, map_path, args.suggest)
    offset = parse_duration(args.duree)
    dd_threshold = abs(args.drawdown)

    end = pd.Timestamp.today().normalize()
    window_start = end - offset
    col = "DD actuel %" if args.mode == "actuel" else "DD max %"
    verbose = args.verbose
    if verbose:
        print(f"{len(uni)} titres, fenêtre depuis {window_start.date()}, "
              f"drawdown {args.mode} <= -{dd_threshold:g} %. Téléchargement…", file=sys.stderr)

    def progress(p):
        print(f"\r  {p:4.0%}", end="", file=sys.stderr, flush=True)

    def pe_progress(p):
        print(f"\r  PER {p:4.0%}", end="", file=sys.stderr, flush=True)

    def roic_progress(p):
        print(f"\r  ROIC {p:4.0%}", end="", file=sys.stderr, flush=True)

    res, _, missing, analysed = ds.run_scan(
        uni, window_start, end, dd_threshold, col, args.above_ma200,
        progress if verbose else None,
        pe_max=args.pe_max, pe_min=args.pe_min,
        pe_col="PER" if args.pe_type == "trailing" else "PER prévu",
        keep_missing_pe=args.keep_no_pe, show_pe=args.show_pe,
        pe_progress_cb=pe_progress if verbose else None,
        roic_min=args.roic_min,
        roic_col="ROIC moy %" if args.roic_type == "avg" else "ROIC %",
        keep_missing_roic=args.keep_no_roic, show_roic=args.show_roic,
        roic_progress_cb=roic_progress if verbose else None)
    if verbose:
        print(file=sys.stderr)
        if ds.CACHE and ds.CACHE.stats:
            st = ds.CACHE.stats
            print(f"Cache : {st['fresh']} à jour, {st['updated']} complétés, "
                  f"{st['full'] + st['redo']} téléchargés en entier, {st['dead']} ignorés "
                  f"(sans cours récemment), {st['missing']} sans réponse.", file=sys.stderr)
        print(f"{len(res)} titres retenus sur {analysed} analysés "
              f"({len(missing)} sans données)\n")
    total = analysed + len(missing)
    if missing:
        if total >= 40 and len(missing) > 0.25 * total:
            # seul message conservé hors mode verbose : la liste serait trompeuse
            print(f"ATTENTION : {len(missing)} titres sur {total} sans données, probablement un "
                  "problème réseau : liste incomplète, relance (-v pour le détail).",
                  file=sys.stderr)
        elif verbose:
            print("Astuce : `--resolve` peut corriger la conversion XTB -> Yahoo "
                  "des titres sans données.", file=sys.stderr)
    if res.empty:
        return 0

    cols = ["ticker", "desc", "currency", "DD actuel %", "DD max %", "Perf période %",
            "vs MM200 %"]
    cols += [c for c in ("PER", "PER prévu", "ROIC %", "ROIC moy %") if c in res.columns]
    cols += ["Dernier cours", "Date plus haut"]
    table = res[cols].rename(columns={"ticker": "Ticker", "desc": "Nom", "currency": "Devise"})
    num = table.select_dtypes("number").columns
    table[num] = table[num].round(2)
    table["Nom"] = table["Nom"].str.slice(0, 32)
    if verbose:
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(table.to_string(index=False))
    else:
        width = res["ticker"].str.len().max()
        print("\n".join(f"{t:<{width}}  {d}" for t, d in zip(res["ticker"], res["desc"])))
    if args.out:
        table.to_csv(args.out, index=False)
        if verbose:
            print(f"\nExporté : {args.out}")
    return 0


if __name__ == "__main__":
    try:
        target = dump_for_llm()
        if any(a in ("-v", "--verbose") for a in sys.argv[1:]):
            print(f"Projet exporté : {target}", file=sys.stderr)
    except Exception as exc:  # l'export ne doit jamais bloquer le scan
        print(f"Export forllm.txt impossible : {exc}", file=sys.stderr)
    sys.exit(main())