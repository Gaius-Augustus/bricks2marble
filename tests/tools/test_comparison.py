"""Tests for :mod:`bricks2marble.tools.comparison`.

The binning helpers are tested directly (no external tool needed). The
end-to-end ``compare(..., track_bins=...)`` test is skipped when the
``gffcompare`` executable is not available.
"""

import shutil
from pathlib import Path

import pytest

from bricks2marble.tools.comparison import (_bin_column, _place_query,
                                            _track_class_codes, compare)

# The header gffcompare writes as the first line of every .tmap file.
_TMAP_HEADER = (
    "ref_gene_id\tref_id\tclass_code\tqry_gene_id\tqry_id\tnum_exons"
    "\tFPKM\tTPM\tcov\tlen\tmajor_iso_id\tref_match_len"
)


def _write_tmap(path, rows):
    """Write a tmap file. Each row is (class_code, num_exons, len)."""
    lines = [_TMAP_HEADER]
    for i, (code, num_exons, length) in enumerate(rows):
        lines.append("\t".join([
            f"r{i}", f"r{i}.t1", code, f"q{i}", f"q{i}.t1",
            str(num_exons), "0.0", "0.0", "0.0", str(length),
            f"q{i}.t1", str(length),
        ]))
    path.write_text("\n".join(lines) + "\n")
    return path


# -------------------------------------------------------------------- #
# _bin_column
# -------------------------------------------------------------------- #
def test_bin_column_integer_edges_and_last_bin_inclusive():
    pairs = [("=", 0), ("=", 300), ("=", 150), ("c", 99)]
    result = _bin_column(pairs, 3)
    # Edges 0/100/200/300; the maximum (300) falls in the last bin.
    assert result["="] == [(0, 100, 1), (100, 200, 1), (200, 300, 1)]
    assert result["c"] == [(0, 100, 1), (100, 200, 0), (200, 300, 0)]
    # Edges are ints because every value is integral.
    assert all(isinstance(lo, int) for lo, _, _ in result["="])


def test_bin_column_float_edges():
    result = _bin_column([("=", 0.5), ("=", 1.5), ("c", 2.5)], 2)
    assert result["="] == [(0.5, 1.5, 1), (1.5, 2.5, 1)]
    assert result["c"] == [(0.5, 1.5, 0), (1.5, 2.5, 1)]


def test_bin_column_left_closed_right_open():
    # A value on an interior edge belongs to the upper bin.
    result = _bin_column([("=", 0), ("=", 100), ("=", 199)], 2)
    # Edges 0/100/199 (rounded); 100 starts the second bin.
    assert result["="][0][2] == 1                  # only 0 in the first bin
    assert result["="][1][2] == 2                  # 100 and 199 in the second


def test_bin_column_empty():
    assert _bin_column([], 4) == {}


def test_bin_column_separates_class_codes():
    result = _bin_column([("=", 10), ("c", 20), ("=", 30)], 1)
    assert set(result) == {"=", "c"}
    assert result["="] == [(10, 30, 2)]
    assert result["c"] == [(10, 30, 1)]


# -------------------------------------------------------------------- #
# _place_query
# -------------------------------------------------------------------- #
def test_place_query_uses_symlink(tmp_path):
    src = tmp_path / "src.gtf"
    src.write_text("hello")
    dst = tmp_path / "cache" / "src.gtf"
    dst.parent.mkdir()

    _place_query(src, dst)
    assert dst.is_symlink()
    assert dst.read_text() == "hello"


def test_place_query_falls_back_to_copy_on_oserror(tmp_path, monkeypatch):
    src = tmp_path / "src.gtf"
    src.write_text("hello")
    dst = tmp_path / "cache" / "src.gtf"
    dst.parent.mkdir()

    # Simulate a filesystem that rejects symlinks (as on some clusters).
    def no_symlink(self, target, target_is_directory=False):
        raise OSError("symlinks not supported")
    monkeypatch.setattr(Path, "symlink_to", no_symlink)

    _place_query(src, dst)
    assert dst.exists()
    assert not dst.is_symlink()          # a real copy, not a link
    assert dst.read_text() == "hello"


# -------------------------------------------------------------------- #
# _track_class_codes
# -------------------------------------------------------------------- #
def test_track_single_column_returns_flat_dict(tmp_path):
    tmap = _write_tmap(tmp_path / "a.tmap", [
        ("=", 2, 100), ("=", 3, 200), ("c", 1, 300), ("=", 2, 300),
    ])
    result = _track_class_codes(tmap, ["Length"], 2)
    # A single column yields {class_code: bins}, not a nested dict.
    assert result == {
        "=": [(100, 200, 1), (200, 300, 2)],
        "c": [(100, 200, 0), (200, 300, 1)],
    }


def test_track_multiple_columns_returns_nested_dict(tmp_path):
    tmap = _write_tmap(tmp_path / "a.tmap", [
        ("=", 2, 100), ("=", 3, 200), ("c", 1, 300),
    ])
    result = _track_class_codes(tmap, ["Length", "Number of exons"], 2)
    assert set(result) == {"Length", "Number of exons"}
    assert result["Length"]["="] == [(100, 200, 1), (200, 300, 1)]
    assert result["Number of exons"]["c"] == [(1, 2, 1), (2, 3, 0)]


def test_track_accepts_raw_header_names(tmp_path):
    tmap = _write_tmap(tmp_path / "a.tmap", [("=", 2, 100), ("c", 1, 300)])
    friendly = _track_class_codes(tmap, ["Length"], 2)
    raw = _track_class_codes(tmap, ["len"], 2)
    assert friendly == raw


def test_track_unknown_column_raises(tmp_path):
    tmap = _write_tmap(tmp_path / "a.tmap", [("=", 2, 100)])
    with pytest.raises(KeyError):
        _track_class_codes(tmap, ["Nonexistent"], 2)


def test_track_counts_cover_every_row(tmp_path):
    rows = [("=", 2, 100), ("=", 3, 250), ("c", 1, 300), ("x", 4, 175)]
    tmap = _write_tmap(tmp_path / "a.tmap", rows)
    result = _track_class_codes(tmap, ["Length"], 3)
    total = sum(n for bins in result.values() for _, _, n in bins)
    assert total == len(rows)


# -------------------------------------------------------------------- #
# End-to-end (requires gffcompare)
# -------------------------------------------------------------------- #
def _gffcompare_available():
    if shutil.which("gffcompare") is not None:
        return True
    try:
        from bricks2marble.tools.external import get_tool_path
        get_tool_path("gffcompare")
        return True
    except Exception:
        return False


def _transcript(gid, exons):
    lo = min(a for a, _ in exons)
    hi = max(b for _, b in exons)
    lines = [
        f'chr1\tt\ttranscript\t{lo}\t{hi}\t.\t+\t.\t'
        f'gene_id "{gid}"; transcript_id "{gid}.t1";'
    ]
    for a, b in exons:
        lines.append(
            f'chr1\tt\texon\t{a}\t{b}\t.\t+\t.\t'
            f'gene_id "{gid}"; transcript_id "{gid}.t1";'
        )
    return "\n".join(lines)


@pytest.mark.skipif(
    not _gffcompare_available(), reason="gffcompare not installed",
)
def test_compare_track_bins_end_to_end(tmp_path):
    ref = tmp_path / "ref.gtf"
    qry = tmp_path / "qry.gtf"
    ref.write_text("\n".join([
        _transcript("r1", [(100, 500)]),
        _transcript("r2", [(1000, 1300)]),
    ]) + "\n")
    qry.write_text("\n".join([
        _transcript("q1", [(100, 500)]),        # matches r1
        _transcript("q2", [(1000, 1250)]),      # overlaps r2
        _transcript("q3", [(5000, 5100)]),      # novel
    ]) + "\n")

    result = compare(qry, ref, track_bins=2)
    # Every query transcript is one tmap row, so the counts sum to 3.
    total = sum(n for bins in result.track.values() for _, _, n in bins)
    assert total == 3
    # Metrics are still parsed alongside the tracking.
    assert result.annotation_loci == 3

    # A plain call leaves the tracking empty and does not pollute the
    # query's directory with .tmap/.refmap files.
    assert compare(qry, ref).track is None
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.suffix in (".tmap", ".refmap")
    ]
    assert leftovers == []


@pytest.mark.skipif(
    not _gffcompare_available(), reason="gffcompare not installed",
)
def test_compare_track_bins_copy_fallback(tmp_path, monkeypatch):
    # Force the symlink to fail, as on a filesystem without symlink
    # support, and check compare still tracks correctly via the copy.
    def no_symlink(self, target, target_is_directory=False):
        raise OSError("symlinks not supported")
    monkeypatch.setattr(Path, "symlink_to", no_symlink)

    ref = tmp_path / "ref.gtf"
    qry = tmp_path / "qry.gtf"
    ref.write_text(_transcript("r1", [(100, 500)]) + "\n")
    qry.write_text("\n".join([
        _transcript("q1", [(100, 500)]),
        _transcript("q2", [(5000, 5100)]),
    ]) + "\n")

    result = compare(qry, ref, track_bins=2)
    total = sum(n for bins in result.track.values() for _, _, n in bins)
    assert total == 2
    # The fallback copy still lands in the temp cache dir, not next to
    # the query.
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.suffix in (".tmap", ".refmap")
    ]
    assert leftovers == []
