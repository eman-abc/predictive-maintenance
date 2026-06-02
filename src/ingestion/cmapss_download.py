"""Download NASA CMAPSS raw files into data/raw/cmapss/."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from src.ingestion.cmapss_config import CMAPSS_DATASET_IDS

# Community mirrors (NASA zip URLs change; raw GitHub mirrors vary by repo layout).
CMAPSS_MIRROR_BASES: tuple[str, ...] = (
    "https://raw.githubusercontent.com/egehanyorulmaz/nasa-turbofan-engine-rul-prediction/main/data",
    # Legacy — often 404; kept as last resort for older notebooks
    "https://raw.githubusercontent.com/kpzhang93/DTAFM/master/CMAPSSData",
)

FILE_STEMS: tuple[str, ...] = tuple(
    f"{prefix}_{fd}"
    for fd in CMAPSS_DATASET_IDS
    for prefix in ("train", "test", "RUL")
)


def _download_one(url: str, dest: Path, *, min_bytes: int = 100) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as exc:
        raise OSError(f"HTTP {exc.code} for {url}") from exc
    if not dest.exists() or dest.stat().st_size < min_bytes:
        dest.unlink(missing_ok=True)
        raise OSError(f"Download too small or missing: {url}")


def download_cmapss_raw(
    raw_dir: str | Path = "data/raw/cmapss",
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Fetch all train/test/RUL text files for FD001–FD004.

    Tries each mirror base until all files are present.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for stem in FILE_STEMS:
        dest = raw_dir / f"{stem}.txt"
        if not force and dest.exists() and dest.stat().st_size >= 100:
            if verbose:
                print(f"ok  {dest.name} ({dest.stat().st_size:,} bytes)")
            continue
        pending.append((stem, dest))

    if not pending:
        return raw_dir

    last_error: str | None = None
    for base in CMAPSS_MIRROR_BASES:
        if verbose:
            print(f"Mirror: {base}")
        errors = 0
        for stem, dest in pending[:]:
            url = f"{base}/{stem}.txt"
            try:
                if verbose:
                    print(f"  get {dest.name}")
                _download_one(url, dest)
                pending.remove((stem, dest))
            except OSError as exc:
                errors += 1
                last_error = str(exc)
                dest.unlink(missing_ok=True)
        if not pending:
            return raw_dir
        if verbose and errors:
            print(f"  {errors} failed on this mirror; trying next…")

    manual = (
        "Could not download all CMAPSS files.\n"
        "Manual options:\n"
        "  1. NASA Prognostics Data Repository (Turbofan Engine Degradation)\n"
        "     https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data\n"
        "  2. Copy train_*.txt, test_*.txt, RUL_*.txt into data/raw/cmapss/\n"
        f"Missing: {[s for s, _ in pending]}\n"
        f"Last error: {last_error}"
    )
    raise FileNotFoundError(manual)


if __name__ == "__main__":
    download_cmapss_raw()
