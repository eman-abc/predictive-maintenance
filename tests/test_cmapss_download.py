"""Tests for CMAPSS download helpers."""

from src.ingestion.cmapss_download import CMAPSS_MIRROR_BASES, FILE_STEMS


def test_mirror_list_has_working_primary():
    assert "egehanyorulmaz" in CMAPSS_MIRROR_BASES[0]


def test_file_stems_cover_all_subsets():
    assert len(FILE_STEMS) == 12
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        assert f"train_{fd}" in FILE_STEMS
        assert f"test_{fd}" in FILE_STEMS
        assert f"RUL_{fd}" in FILE_STEMS
