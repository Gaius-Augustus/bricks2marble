"""Tests for :mod:`bricks2marble.tools.annotate`."""

import numpy as np
import pytest

from bricks2marble.struct import Fasta, Sequence
from bricks2marble.tools import annotate_genome
from bricks2marble.tools.annotate import (Region, _annotate,
                                          _annotation_from_dict,
                                          _find_mismatches, _first_non_zero,
                                          _merge_replace_center,
                                          _split_regions,
                                          _transcripts_from_regions)

_HMM_TRANSITIONS = {
    (0, 0), (1, 1), (2, 2), (3, 3),          # self loops
    (0, 7),                                  # IR -> START
    (1, 11), (2, 12), (3, 13),               # Ik -> IEk
    (8, 1), (9, 2), (10, 3),                 # EIk -> Ik
    (7, 5), (5, 14), (14, 0),                # START -> E1 -> STOP -> IR
    (4, 5), (5, 6), (6, 4),                  # E0 -> E1 -> E2 -> E0
    (4, 8), (5, 9), (6, 10),                 # Ek -> EIk
    (11, 4), (12, 5), (13, 6),               # IEk -> Ek
}


def _assert_valid_hmm_path(states, breaks=()):
    """Assert that ``states`` only uses transitions the HMM allows.

    ``breaks`` lists indices whose outgoing transition may be invalid,
    used for predictions that are deliberately wrong where two chunks
    meet.
    """
    states = np.asarray(states)
    bad = [
        (i, (int(states[i]), int(states[i+1])))
        for i in range(states.size - 1)
        if (int(states[i]), int(states[i+1])) not in _HMM_TRANSITIONS
        and i not in breaks
    ]
    assert bad == [], f"invalid HMM transitions at {bad}"


# -------------------------------------------------------------------- #
# helpers
# -------------------------------------------------------------------- #
def _regions(*blocks, strand="+"):
    """Build a CDS/intron region list from ``(start, end)`` blocks."""
    out = []
    for i, (s, e) in enumerate(blocks):
        if i:
            out.append(Region(
                name="intron", start=blocks[i-1][1], end=s, strand=strand,
            ))
        out.append(Region(name="CDS", start=s, end=e, strand=strand))
    return out


def _genes(annotation):
    """Compact view of an annotation for comparison in tests:
    ``(sequence, strand, [[(cds_start, cds_end), ...], ...])`` per gene.
    """
    return [
        (
            gene.sequence,
            gene.strand,
            [[(c.start, c.end) for c in tx.cds] for tx in gene.transcripts],
        )
        for gene in annotation.genes()
    ]


def _features(path, feature):
    """Return the split rows of a written file matching a feature."""
    return [
        ln.split("\t") for ln in path.read_text().splitlines()
        if not ln.startswith("#") and ln.split("\t")[2] == feature
    ]


# -------------------------------------------------------------------- #
# _split_regions
# -------------------------------------------------------------------- #
def test_split_regions_aggregates_hmm_states():
    # IR -> START E1 E2 E0 -> EI0 -> I0* -> IE0 -> E0 E1 -> STOP -> IR.
    # The borders (8, 11), START and STOP all aggregate to "CDS", so only
    # the looping intron states show up as an intron.
    states = np.array(
        3*[0] + [7, 5, 6, 4] + [8] + 4*[1] + [11] + [4, 5] + [14] + 3*[0]
    )
    _assert_valid_hmm_path(states)
    regions = _split_regions(states)
    assert [(r.name, r.start, r.end) for r in regions] == [
        ("intergenic", 0, 3),
        ("CDS", 3, 8),
        ("intron", 8, 12),
        ("CDS", 12, 16),
        ("intergenic", 16, 19),
    ]
    assert all(r.strand == "+" for r in regions)


def test_split_regions_applies_offset_and_strand():
    # IR -> START E1 E2 E0 E1 -> STOP -> IR (single coding block).
    states = np.array(3*[0] + [7, 5, 6, 4, 5, 14] + 3*[0])
    _assert_valid_hmm_path(states)
    regions = _split_regions(states, offset=100, strand="-")
    assert [(r.name, r.start, r.end) for r in regions] == [
        ("intergenic", 100, 103),
        ("CDS", 103, 109),
        ("intergenic", 109, 112),
    ]
    assert all(r.strand == "-" for r in regions)


# -------------------------------------------------------------------- #
# _transcripts_from_regions
# -------------------------------------------------------------------- #
def test_transcripts_from_regions_keeps_complete_transcript():
    states = np.array(
        3*[0] + [7, 5, 6, 4] + [8] + 4*[1] + [11] + [4, 5] + [14] + 3*[0]
    )
    _assert_valid_hmm_path(states)
    txs = _transcripts_from_regions(_split_regions(states))
    assert len(txs) == 1
    assert [(r.name, r.start, r.end) for r in txs[0]] == [
        ("CDS", 3, 8), ("intron", 8, 12), ("CDS", 12, 16),
    ]


def test_transcripts_from_regions_drops_leading_partial():
    # The chunk starts inside a gene (mid coding cycle), so that first
    # transcript has no intergenic region in front of it and cannot be
    # trusted; only the fully contained gene is kept.
    states = np.array([4, 5, 14] + 3*[0] + [7, 5, 6, 4, 5, 14] + 3*[0])
    _assert_valid_hmm_path(states)
    txs = _transcripts_from_regions(_split_regions(states))
    assert [[(r.name, r.start, r.end) for r in t] for t in txs] == [
        [("CDS", 6, 12)],
    ]


# -------------------------------------------------------------------- #
# _find_mismatches
# -------------------------------------------------------------------- #
def test_find_mismatches_none_when_chunks_meet_in_intergenic():
    assert _find_mismatches(np.zeros((3, 4), dtype=int)).size == 0


def test_find_mismatches_on_non_intergenic_boundary():
    # Chunk 1 ends in an exon state, so the boundary is unreliable even
    # though both chunks agree on the state.
    pred = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 5],
        [5, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert _find_mismatches(pred).tolist() == [1]


def test_find_mismatches_on_state_disagreement():
    pred = np.array([[0, 0, 0, 1], [2, 0, 0, 0]])
    assert _find_mismatches(pred).tolist() == [0]


def test_find_mismatches_exon_at_boundary():
    pred = np.array([[0, 0, 4, 0], [0, 0, 0, 0]])
    assert _find_mismatches(pred).size == 0
    assert _find_mismatches(pred, exon_at_boundary=2).tolist() == [0]


# -------------------------------------------------------------------- #
# _merge_replace_center / _first_non_zero
# -------------------------------------------------------------------- #
def test_merge_replace_center_success():
    new_left, new_right, ok = _merge_replace_center(
        np.array([0, 0, 1, 2]), np.array([3, 4, 0, 0]),
        np.array([1, 2, 3, 4]),
    )
    assert ok
    assert new_left.tolist() == [0, 0, 1, 2]
    assert new_right.tolist() == [3, 4, 0, 0]


def test_merge_replace_center_failure_leaves_input_untouched():
    new_left, new_right, ok = _merge_replace_center(
        np.array([9, 9, 9, 9]), np.array([9, 9, 9, 9]),
        np.array([1, 2, 3, 4]),
    )
    assert not ok
    assert new_left.tolist() == [9, 9, 9, 9]
    assert new_right.tolist() == [9, 9, 9, 9]


def test_first_non_zero():
    assert _first_non_zero(np.array([False, True, True])) == 1
    assert _first_non_zero(np.array([0, 0, 3, 4])) == 2
    assert _first_non_zero(np.array([0, 0])) == -1
    assert _first_non_zero(np.array([])) == -1


# -------------------------------------------------------------------- #
# _annotation_from_dict
# -------------------------------------------------------------------- #
def test_annotation_from_dict_interleaves_strands_by_start():
    fwd = {"chr1": [_regions((10, 20), (30, 40)), _regions((100, 130))]}
    bwd = {"chr1": [_regions((50, 60), strand="-")]}
    assert _genes(_annotation_from_dict(fwd, bwd)) == [
        ("chr1", "+", [[(10, 20), (30, 40)]]),
        ("chr1", "-", [[(50, 60)]]),
        ("chr1", "+", [[(100, 130)]]),
    ]


def test_annotation_from_dict_numbers_genes_from_one(tmp_path):
    ann = _annotation_from_dict(
        {"chr1": [_regions((10, 20)), _regions((30, 40))]}, {},
    )
    assert ann.next_gene_id == 3
    out = tmp_path / "o.gtf"
    ann.to_gtf(out)
    assert [f[8] for f in _features(out, "gene")] == [
        'gene_id "g1";', 'gene_id "g2";',
    ]


def test_annotation_from_dict_continues_numbering(tmp_path):
    first = _annotation_from_dict({"chr1": [_regions((10, 20))]}, {})
    second = _annotation_from_dict(
        {"chr2": [_regions((5, 15))]}, {}, first_tx_id=first.next_gene_id,
    )
    out = tmp_path / "o.gtf"
    first.to_gtf(out)
    second.to_gtf(out, append=True)
    assert [f[8] for f in _features(out, "gene")] == [
        'gene_id "g1";', 'gene_id "g2";',
    ]


# -------------------------------------------------------------------- #
# A genome of three sequences, predicted on both strands
# -------------------------------------------------------------------- #
# a 3000 nt genome split into three sequences of 1000. Each block below
# is one sequence and is a valid path through the HMM (asserted further
# down).
EXAMPLE_FWD = np.array([]
    # seq_0: IR - exon - I0 - exon - IR
    + 100*[0] + [7] + 30*[5, 6, 4] + [8] + 200*[1] + [11] + 30*[4, 5, 6]
    + [4, 5, 14] + 514*[0]

    # seq_1: IR - exon - I0 - exon - I2 - exon - IR
    + 50*[0] + [7] + 10*[5, 6, 4] + [8] + 100*[1] + [11] + 10*[4, 5, 6]
    + [10] + 100*[3] + [13] + 10*[6, 4, 5] + [14] + 654*[0]

    # seq_2: no gene on this strand
    + 1000*[0]
)
EXAMPLE_BWD = np.array([]
    # seq_0: IR - exon - IR (a single coding block)
    + 150*[0] + [7] + 33*[5, 6, 4] + [5, 14] + 748*[0]

    # seq_1: no gene on this strand
    + 1000*[0]

    # seq_2: IR - exon - I0 - exon - IR
    + 200*[0] + [7] + 20*[5, 6, 4] + [8] + 150*[1] + [11] + 20*[4, 5, 6]
    + [4, 5, 14] + 524*[0]
)


def _example_fasta():
    """Three sequences of length 1000 covering a 3000 nt genome. The
    nucleotides themselves never influence the annotation, only the
    predicted states do.
    """
    return Fasta([
        Sequence(
            np.zeros(1000, dtype=np.int8),
            name=f"seq_{i}",
            start=i*1000,
            end=(i+1)*1000,
        )
        for i in range(3)
    ]).resample(1000)


def _example_predict(fasta):
    """Pseudo model returning the precomputed states per sequence."""
    fwd, bwd = [], []
    for seq in fasta:
        for _ in seq.nuc:
            fwd.append(EXAMPLE_FWD[seq.start:seq.end])
            bwd.append(EXAMPLE_BWD[seq.start:seq.end])
    return np.array(fwd), np.array(bwd)


def test_example_state_arrays_cover_the_genome():
    assert EXAMPLE_FWD.shape == (3000,)
    assert EXAMPLE_BWD.shape == (3000,)


def test_example_state_arrays_are_valid_hmm_paths():
    # Each sequence is predicted on its own, so every 1000 nt block has
    # to be a path the HMM can actually walk.
    for i in range(3):
        _assert_valid_hmm_path(EXAMPLE_FWD[i*1000:(i+1)*1000])
        _assert_valid_hmm_path(EXAMPLE_BWD[i*1000:(i+1)*1000])


def test_annotate_three_sequences():
    ann = _annotate(_example_fasta(), predict_func=_example_predict)
    assert ann.sequences() == ["seq_0", "seq_1", "seq_2"]
    # Coordinates are 0-based and relative to each sequence. The coding
    # regions run from START up to and including the border state, and
    # resume at the border state on the other side of the intron.
    assert _genes(ann) == [
        ("seq_0", "+", [[(100, 192), (392, 486)]]),
        ("seq_0", "-", [[(150, 252)]]),
        ("seq_1", "+", [[(50, 82), (182, 214), (314, 346)]]),
        ("seq_2", "-", [[(200, 262), (412, 476)]]),
    ]


def test_annotate_single_strand_only():
    """A predict function may leave one strand out entirely."""
    def predict_fwd_only(fasta):
        fwd, _ = _example_predict(fasta)
        return fwd, None

    ann = _annotate(_example_fasta(), predict_func=predict_fwd_only)
    assert {strand for _, strand, _ in _genes(ann)} == {"+"}


# -------------------------------------------------------------------- #
# example 2: repredicting failed chunk boundaries
# -------------------------------------------------------------------- #
PREDICT_FWD = np.array([
    0,  0,  0,  0,  0,  0,  7,  5,  9,  2,
    2,  2, 12,  5,  6,  4,  5, 14,  0,  0,  # fail
    6,  4,  5,  6,  4,  5,  6, 10,  3,  3,  # fail
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  # fail
    2,  2,  2,  2,  2, 12,  5, 14,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
])
PREDICT_BWD = np.zeros(70, dtype=int)
REPREDICT_FWD = np.array([
    4,  5, 14,  0,  0,  0,  7,  5,  6,  4,
    5, 14,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  7,  5,  6,  4,  5,  9,  2,  2,
])
REPREDICT_BWD = np.zeros(30, dtype=int)


def _repredict_fasta():
    return Fasta([
        Sequence(np.zeros(70, dtype=np.int8), name="chr1"),
    ]).resample(10)


def test_predict_states_only_break_where_chunks_meet():
    # The prediction is a valid HMM path inside every chunk and is only
    # wrong where two chunks meet (the last index of chunks 1, 2 and 4).
    # Those are exactly the boundaries _find_mismatches has to report.
    _assert_valid_hmm_path(PREDICT_FWD, breaks={19, 29, 49})
    _assert_valid_hmm_path(PREDICT_BWD)
    _assert_valid_hmm_path(REPREDICT_FWD)
    _assert_valid_hmm_path(REPREDICT_BWD)


def test_find_mismatches_of_the_notebook_prediction():
    assert _find_mismatches(PREDICT_FWD.reshape(7, 10)).tolist() == [1, 2, 4]
    assert _find_mismatches(PREDICT_BWD.reshape(7, 10)).size == 0


def test_annotate_repredicts_failing_boundaries():
    calls = []

    def predict_func(fasta):
        # Copy: _annotate merges repredictions into these arrays.
        return PREDICT_FWD.reshape(7, 10).copy(), \
            PREDICT_BWD.reshape(7, 10).copy()

    def repredict_func(fasta):
        calls.append((fasta.N, fasta.T))
        return REPREDICT_FWD.reshape(3, 10).copy(), \
            REPREDICT_BWD.reshape(3, 10).copy()

    ann = _annotate(
        _repredict_fasta(),
        predict_func=predict_func,
        repredict_func=repredict_func,
    )
    # The three mismatching boundaries are handed to repredict_func as
    # one Fasta of three chunks.
    assert calls == [(3, 10)]
    assert _genes(ann) == [
        ("chr1", "+", [[(6, 9), (12, 18)]]),
        ("chr1", "+", [[(21, 27)]]),
        ("chr1", "+", [[(47, 53), (55, 58)]]),
    ]


def test_repredicted_annotation_gtf_matches_notebook(tmp_path):
    def predict_func(fasta):
        return PREDICT_FWD.reshape(7, 10).copy(), \
            PREDICT_BWD.reshape(7, 10).copy()

    def repredict_func(fasta):
        return REPREDICT_FWD.reshape(3, 10).copy(), \
            REPREDICT_BWD.reshape(3, 10).copy()

    ann = _annotate(
        _repredict_fasta(),
        predict_func=predict_func,
        repredict_func=repredict_func,
    )
    out = tmp_path / "o.gtf"
    ann.to_gtf(out, source="Exampler")

    # 1-based fully closed coordinates
    assert [(f[3], f[4], f[8]) for f in _features(out, "gene")] == [
        ("7", "18", 'gene_id "g1";'),
        ("22", "27", 'gene_id "g2";'),
        ("48", "58", 'gene_id "g3";'),
    ]
    assert [(f[3], f[4]) for f in _features(out, "CDS")] == [
        ("7", "9"), ("13", "18"),
        ("22", "27"),
        ("48", "53"), ("56", "58"),
    ]
    assert [(f[3], f[4]) for f in _features(out, "intron")] == [
        ("10", "12"), ("54", "55"),
    ]
    assert [(f[3], f[4]) for f in _features(out, "start_codon")] == [
        ("7", "9"), ("22", "24"), ("48", "50"),
    ]
    assert [(f[3], f[4]) for f in _features(out, "stop_codon")] == [
        ("16", "18"), ("25", "27"), ("56", "58"),
    ]
    assert all(f[1] == "Exampler" for f in _features(out, "CDS"))


# -------------------------------------------------------------------- #
# annotate_genome (file in, file out)
# -------------------------------------------------------------------- #
def _write_fasta(path, name="chr1", length=100):
    path.write_text(f">{name}\n" + "ACGT" * (length // 4) + "\n")
    return path


def _one_gene_predict(fasta):
    """Predict one gene at a fixed offset in every chunk:
    IR -> START -> E1 E2 E0 E1 E2 E0 E1 -> STOP -> IR. The coding run
    has to end on E1, since that is the only state STOP can be entered
    from.
    """
    fwd = np.zeros((fasta.N, fasta.T), dtype=int)
    fwd[:, 10] = 7                                    # IR -> START
    fwd[:, 11:18] = np.r_[np.tile([5, 6, 4], 2), 5]   # E1 E2 E0 ... E1
    fwd[:, 18] = 14                                   # E1 -> STOP
    return fwd, np.zeros((fasta.N, fasta.T), dtype=int)


def test_one_gene_prediction_is_a_valid_hmm_path():
    fwd, bwd = _one_gene_predict(Fasta([
        Sequence(np.zeros(100, dtype=np.int8), name="chr1"),
    ]).resample(100))
    _assert_valid_hmm_path(fwd[0])
    _assert_valid_hmm_path(bwd[0])


def test_annotate_genome_writes_gtf(tmp_path):
    out = tmp_path / "out.gtf"
    annotate_genome(
        fasta=_write_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        model_name="Exampler",
        T_max=100,
        min_sequence_size=None,
    )
    genes = _features(out, "gene")
    assert len(genes) == 1
    assert genes[0][0] == "chr1"
    assert genes[0][1] == "Exampler"           # source is the model name
    assert (genes[0][3], genes[0][4]) == ("11", "19")
    assert genes[0][8] == 'gene_id "g1";'


def test_annotate_genome_writes_log_next_to_output(tmp_path):
    out = tmp_path / "out.gtf"
    annotate_genome(
        fasta=_write_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        T_max=100,
        min_sequence_size=None,
    )
    assert (tmp_path / "out.log").exists()


def test_annotate_genome_writes_gff3(tmp_path):
    out = tmp_path / "out.gff3"
    annotate_genome(
        fasta=_write_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        T_max=100,
        min_sequence_size=None,
    )
    text = out.read_text()
    assert text.splitlines()[0] == "##gff-version 3"
    assert [f[8] for f in _features(out, "gene")] == ["ID=g1;Name=g1"]
    assert [f[8] for f in _features(out, "mRNA")] == [
        "ID=g1_t1;Name=g1_t1;Parent=g1",
    ]


def test_annotate_genome_refuses_existing_output(tmp_path):
    out = tmp_path / "out.gtf"
    out.write_text("")
    with pytest.raises(FileExistsError):
        annotate_genome(
            fasta=_write_fasta(tmp_path / "genome.fa"),
            predict_func=_one_gene_predict,
            output=out,
            T_max=100,
            min_sequence_size=None,
        )


def test_annotate_genome_applies_postprocess(tmp_path):
    def drop_all(group, annotation):
        for gene in list(annotation.genes()):
            annotation.remove(gene)
        return annotation

    out = tmp_path / "out.gtf"
    annotate_genome(
        fasta=_write_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        T_max=100,
        min_sequence_size=None,
        postprocess=drop_all,
    )
    assert _features(out, "gene") == []


def _two_sequence_fasta(path):
    path.write_text(
        ">chr1\n" + "ACGT" * 25 + "\n>chr2\n" + "ACGT" * 25 + "\n"
    )
    return path


def test_annotate_genome_numbers_genes_across_sequences(tmp_path):
    out = tmp_path / "out.gtf"
    annotate_genome(
        fasta=_two_sequence_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        T_max=100,
        min_sequence_size=None,
    )
    assert [(f[0], f[8]) for f in _features(out, "gene")] == [
        ("chr1", 'gene_id "g1";'),
        ("chr2", 'gene_id "g2";'),
    ]


def test_annotate_genome_excludes_sequences(tmp_path):
    out = tmp_path / "out.gtf"
    annotate_genome(
        fasta=_two_sequence_fasta(tmp_path / "genome.fa"),
        predict_func=_one_gene_predict,
        output=out,
        T_max=100,
        min_sequence_size=None,
        exclude_seqs=["chr2"],
    )
    # Without the filter both sequences are annotated (see the test
    # above), so this really exercises exclude_seqs.
    assert {f[0] for f in _features(out, "gene")} == {"chr1"}
