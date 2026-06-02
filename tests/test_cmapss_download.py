"""Tests for CMAPSS download/import helpers."""

from pathlib import Path

from src.ingestion.cmapss_download import (
    CMAPSS_MIRROR_BASES,
    FILE_STEMS,
    import_cmapss_from_dir,
    missing_cmapss_files,
)


def test_mirror_list_has_working_primary():
    assert "egehanyorulmaz" in CMAPSS_MIRROR_BASES[0]


def test_file_stems_cover_all_subsets():
    assert len(FILE_STEMS) == 12
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        assert f"train_{fd}" in FILE_STEMS
        assert f"test_{fd}" in FILE_STEMS
        assert f"RUL_{fd}" in FILE_STEMS


def test_import_from_upload_dir(tmp_path):
    upload = tmp_path / "upload"
    raw = tmp_path / "raw"
    upload.mkdir()
    (upload / "train_FD001.txt").write_text("1 1 0 0 0 " + "1 " * 21 + "\n", encoding="utf-8")
    missing = missing_cmapss_files(raw)
    assert "train_FD001.txt" in missing
    try:
        import_cmapss_from_dir(upload, raw, verbose=False)
    except FileNotFoundError:
        pass
    assert (raw / "train_FD001.txt").exists()
