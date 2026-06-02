"""Download or import NASA CMAPSS raw files into data/raw/cmapss/."""

from __future__ import annotations

import shutil
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

REQUIRED_FILENAMES: tuple[str, ...] = tuple(f"{stem}.txt" for stem in FILE_STEMS)


def missing_cmapss_files(raw_dir: str | Path) -> list[str]:
    """Return required filenames not present (or too small) under raw_dir."""
    raw_dir = Path(raw_dir)
    missing = []
    for name in REQUIRED_FILENAMES:
        path = raw_dir / name
        if not path.is_file() or path.stat().st_size < 100:
            missing.append(name)
    return missing


def _find_uploaded_file(source_dir: Path, stem: str) -> Path | None:
    """Locate stem.txt in source_dir or source_dir/CMAPSSData."""
    for candidate in (source_dir / f"{stem}.txt", source_dir / "CMAPSSData" / f"{stem}.txt"):
        if candidate.is_file() and candidate.stat().st_size >= 100:
            return candidate
    return None


def import_cmapss_from_dir(
    source_dir: str | Path,
    raw_dir: str | Path = "data/raw/cmapss",
    *,
    verbose: bool = True,
) -> Path:
    """
    Copy uploaded CMAPSS files from Colab disk (or any folder) into data/raw/cmapss/.

    Accepts flat layout or NASA zip extract folder ``CMAPSSData/``.
    """
    source_dir = Path(source_dir)
    raw_dir = Path(raw_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Upload folder not found: {source_dir.resolve()}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for stem in FILE_STEMS:
        src = _find_uploaded_file(source_dir, stem)
        if src is None:
            continue
        dest = raw_dir / f"{stem}.txt"
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            if verbose:
                print(f"ok  {dest.name} (already in place)")
            continue
        shutil.copy2(src, dest)
        copied += 1
        if verbose:
            print(f"copy {src.name} -> {dest} ({dest.stat().st_size:,} bytes)")

    still_missing = missing_cmapss_files(raw_dir)
    if still_missing:
        raise FileNotFoundError(
            f"Still missing {len(still_missing)} file(s) in {raw_dir}.\n"
            f"Upload to {source_dir}:\n  " + "\n  ".join(still_missing[:6])
            + ("\n  …" if len(still_missing) > 6 else "")
        )
    if verbose:
        print(f"All 12 CMAPSS files ready in {raw_dir.resolve()} ({copied} copied this run)")
    return raw_dir


def ensure_cmapss_raw(
    raw_dir: str | Path = "data/raw/cmapss",
    *,
    upload_dir: str | Path | None = None,
    download: bool = False,
    verbose: bool = True,
) -> Path:
    """Use upload folder and/or download until all 12 raw files exist."""
    raw_dir = Path(raw_dir)
    if not missing_cmapss_files(raw_dir):
        if verbose:
            print(f"CMAPSS raw data already in {raw_dir.resolve()}")
        return raw_dir
    if upload_dir is not None:
        return import_cmapss_from_dir(upload_dir, raw_dir, verbose=verbose)
    if download:
        download_cmapss_raw(raw_dir, verbose=verbose)
        return raw_dir
    raise FileNotFoundError(
        f"CMAPSS files missing in {raw_dir}. "
        "Upload train_*.txt, test_*.txt, RUL_*.txt or pass upload_dir=/content/cmapss_upload"
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
