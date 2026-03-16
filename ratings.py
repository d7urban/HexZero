"""
ratings.py — Display the BOTB.rating rankings table.

Usage:
    python ratings.py
    python ratings.py --all     # include unplayed nets (RD = 350)
"""

import argparse
import hashlib
import json
import os

_RATING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BOTB.rating")
_INITIAL_RD  = 173.7178 * 2.0145  # ≈ 350 — default RD for a net that has never played
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))


def _live_hashes() -> set[str]:
    """Return the SHA-256 of every best.pt that currently exists on disk."""
    hashes: set[str] = set()
    try:
        for name in os.listdir(_BASE_DIR):
            best_pt = os.path.join(_BASE_DIR, name, "best.pt")
            if os.path.isfile(best_pt):
                h = hashlib.sha256()
                with open(best_pt, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                hashes.add(h.hexdigest())
    except OSError:
        pass
    return hashes


def main() -> None:
    p = argparse.ArgumentParser(description="Show Glicko-2 rankings from BOTB.rating")
    p.add_argument("--all", action="store_true",
                   help="Include nets that have never played (RD ≈ 350)")
    p.add_argument("--sort", default="rating",
                   choices=["rating", "rd", "sigma", "label", "id", "live"],
                   help="Sort column (default: rating)")
    args = p.parse_args()

    if not os.path.exists(_RATING_FILE):
        print("No BOTB.rating file found. Run tournament.py first.")
        return

    with open(_RATING_FILE) as f:
        data = json.load(f)

    entries = [
        (key, entry)
        for key, entry in data.items()
        if not key.startswith("_")
    ]

    if not args.all:
        entries = [(k, e) for k, e in entries if e.get("rd", _INITIAL_RD) < _INITIAL_RD - 1]

    if not entries:
        print("No rated nets found. Run tournament.py first.")
        return

    live = _live_hashes()

    sort_key = {
        "rating": lambda x: -x[1].get("rating", 1500),
        "rd":     lambda x:  x[1].get("rd",     _INITIAL_RD),
        "sigma":  lambda x:  x[1].get("sigma",  0.06),
        "label":  lambda x:  x[1].get("label",  "").lower(),
        "id":     lambda x:  int(x[0]),
        "live":   lambda x: (0 if x[1].get("sha256") in live else 1, -x[1].get("rating", 1500)),
    }
    entries.sort(key=sort_key[args.sort])

    col = max(len(e.get("label", "")) for _, e in entries) + 2
    sep = "─" * (col + 36)

    print(sep)
    print(f"  {'#':<5} {'Label':<{col}} {'Rating':>7}  {'RD':>6}  {'σ':>7}  {'':2}")
    print(sep)
    for rank, (key, e) in enumerate(entries, 1):
        marker = "* " if e.get("sha256") in live else "  "
        print(f"  {key:<5} {e.get('label', '?'):<{col}} "
              f"{e.get('rating', 1500):>7.0f}  "
              f"{e.get('rd', _INITIAL_RD):>6.0f}  "
              f"{e.get('sigma', 0.06):>7.4f}  {marker}")
    print(sep)
    print(f"  {len(entries)} net(s) rated  (* = currently on disk)")


if __name__ == "__main__":
    main()
