"""Tests for :mod:`bricks2marble.struct.annotation`."""

import pytest

from bricks2marble.io import fasta_from_string
from bricks2marble.struct.annotation import (CDS, Annotation, Gene,
                                             SequenceAnnotation, Transcript)


# -------------------------------------------------------------------- #
# helpers
# -------------------------------------------------------------------- #
def tx(*blocks, sequence="chr1", strand="+", name=None):
    """Build a transcript from ``(start, end)`` block tuples."""
    return Transcript(
        name=name,
        sequence=sequence,
        strand=strand,
        cds=[CDS(start=s, end=e) for s, e in blocks],
    )


# -------------------------------------------------------------------- #
# CDS / Transcript basics
# -------------------------------------------------------------------- #
def test_cds_equality_and_hash():
    assert CDS(start=1, end=5) == CDS(start=1, end=5)
    assert CDS(start=1, end=5) != CDS(start=1, end=6)
    assert hash(CDS(start=1, end=5)) == hash(CDS(start=1, end=5))


def test_transcript_extent_and_lengths():
    t = tx((10, 20), (30, 45))
    assert t.start == 10
    assert t.end == 45
    assert t.cds_lengths() == [10, 15]
    assert t.cds_length() == 25
    assert t.intron_lengths() == [10]


def test_transcript_classify():
    t = tx((10, 20), (30, 40))
    assert t.classify(9) is None          # before the transcript
    assert t.classify(15) == "cds"
    assert t.classify(25) == "intron"
    assert t.classify(40) is None         # end-exclusive


def test_transcript_equality_ignores_name():
    a = tx((10, 20), name="foo")
    b = tx((10, 20), name="bar")
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_transcript_name_is_optional():
    t = tx((10, 20))
    assert t.name is None
    # Name is generated from the given counters at write time.
    assert t.display_name(3, 1) == "g3_t1"
    # An explicit name always wins over the counters.
    assert tx((10, 20), name="foo").display_name(3, 1) == "foo"


# -------------------------------------------------------------------- #
# Gene: ids, parents, classification, longest
# -------------------------------------------------------------------- #
def test_gene_add_appends_in_order():
    g = Gene(sequence="chr1", strand="+", transcripts=[tx((10, 20))])
    extra = tx((10, 30))
    g.add(extra)
    assert g.transcripts[-1] is extra
    assert len(g.transcripts) == 2


def test_gene_extent_and_longest():
    short = tx((10, 20))
    long = tx((100, 130))
    g = Gene(sequence="chr1", strand="+", transcripts=[short, long])
    assert g.start == 10
    assert g.end == 130
    assert g.longest() is long


def test_gene_classify_cds_precedence():
    # Position 22 is intron in the first isoform but CDS in the second.
    g = Gene(sequence="chr1", strand="+", transcripts=[
        tx((10, 20), (30, 40)),
        tx((10, 25), (30, 40)),
    ])
    assert g.classify(22) == "cds"
    assert g.classify(27) == "intron"
    assert g.classify(50) is None


def test_gene_display_name_uses_explicit_name_or_pattern():
    named = Gene(name="BRCA1", sequence="chr1", strand="+",
                 transcripts=[tx((1, 2))])
    assert named.display_name(7) == "BRCA1"

    auto = Gene(sequence="chr1", strand="+", transcripts=[tx((1, 2))])
    assert auto.display_name(7) == "g7"
    assert auto.display_name(7, "gene{gene_id}") == "gene7"


def test_gene_from_transcript():
    t = tx((10, 20))
    g = Gene.from_transcript(t, name="G")
    assert g.name == "G"
    assert g.sequence == "chr1"
    assert g.strand == "+"
    assert g.transcripts == [t]


# -------------------------------------------------------------------- #
# SequenceAnnotation
# -------------------------------------------------------------------- #
def test_sequence_annotation_is_not_iterable():
    sa = SequenceAnnotation("chr1")
    with pytest.raises(TypeError):
        iter(sa)


def test_sequence_annotation_add_transcript_wraps_in_gene():
    sa = SequenceAnnotation("chr1")
    sa.add(tx((10, 20)))
    sa.add(tx((100, 130)))
    genes = list(sa.genes())
    assert len(genes) == 2
    assert all(g.name is None for g in genes)
    assert len(list(sa.transcripts())) == 2
    assert len(sa) == 2


def test_sequence_annotation_rejects_foreign_sequence():
    sa = SequenceAnnotation("chr1")
    with pytest.raises(ValueError):
        sa.add(tx((10, 20), sequence="chr2"))


def test_sequence_annotation_remove_transcript_and_gene():
    sa = SequenceAnnotation("chr1")
    g = Gene(sequence="chr1", strand="+", transcripts=[
        tx((10, 20)), tx((10, 25)),
    ])
    sa.add(g)
    sa.add(tx((100, 130)))

    sa.remove(g.transcripts[1])            # drop one isoform
    assert len(g.transcripts) == 1
    assert len(list(sa.genes())) == 2      # gene still present

    sa.remove(g)                           # drop the whole gene
    assert len(list(sa.genes())) == 1
    assert len(sa) == 1


def test_sequence_annotation_remove_missing_raises():
    sa = SequenceAnnotation("chr1")
    sa.add(tx((10, 20)))
    with pytest.raises(KeyError):
        sa.remove(tx((999, 1000)))


def test_sequence_annotation_classify_strands():
    sa = SequenceAnnotation("chr1")
    sa.add(tx((10, 20), (30, 40), strand="+"))
    sa.add(tx((100, 120), strand="-"))
    assert sa.classify(15) == ("cds", "intergenic")
    assert sa.classify(25) == ("intron", "intergenic")
    assert sa.classify(110) == ("intergenic", "cds")
    assert sa.classify(200) == ("intergenic", "intergenic")


def test_sequence_annotation_classify_range():
    sa = SequenceAnnotation("chr1")
    sa.add(tx((10, 20), (30, 40), strand="+"))
    fwd = sa._classify_range_strand(0, 45, "+")
    assert fwd["cds"] == [(10, 20), (30, 40)]
    assert fwd["intron"] == [(20, 30)]
    assert fwd["intergenic"] == [(0, 10), (40, 45)]


# -------------------------------------------------------------------- #
# Annotation: structure, ids, naming
# -------------------------------------------------------------------- #
def test_gene_names_follow_write_order(tmp_path):
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chr1"))
    ann.add(tx((10, 20), sequence="chr2"))
    ann.add(tx((30, 40), sequence="chr1"))
    assert ann.sequences() == ["chr1", "chr2"]
    out = tmp_path / "o.gtf"
    ann.to_gtf(out)
    gene_ids = [ln.split("\t")[8] for ln in out.read_text().splitlines()
                if ln.split("\t")[2] == "gene"]
    # Numbered from 1 in write order (chr1 genes first, then chr2).
    assert gene_ids == ['gene_id "g1";', 'gene_id "g2";', 'gene_id "g3";']


def test_removal_leaves_no_numbering_gap(tmp_path):
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chr1"))
    middle = tx((100, 130), sequence="chr1")
    ann.add(middle)
    ann.add(tx((200, 230), sequence="chr1"))
    ann.remove(middle)                       # drop the middle gene
    out = tmp_path / "o.gtf"
    ann.to_gtf(out)
    gene_ids = [ln.split("\t")[8] for ln in out.read_text().splitlines()
                if ln.split("\t")[2] == "gene"]
    # Remaining genes are renumbered contiguously: no gap at g2.
    assert gene_ids == ['gene_id "g1";', 'gene_id "g2";']


def test_annotation_iterates_sequence_annotations():
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chr1"))
    seq_anns = list(ann)
    assert len(seq_anns) == 1
    assert isinstance(seq_anns[0], SequenceAnnotation)
    assert list(ann.genes())[0] is next(seq_anns[0].genes())


def test_annotation_getitem_by_name_and_index():
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chrX"))
    assert ann["chrX"] is ann[0]
    with pytest.raises(KeyError):
        ann["missing"]


def test_gene_numbering_starts_at_one_by_default():
    ann = Annotation()
    assert ann.next_gene_id == 1          # nothing added yet
    ann.add(tx((10, 20), sequence="chr1"))
    assert next(ann.genes()).display_name(ann.next_gene_id - 1) == "g1"


def test_start_gene_id_and_next_gene_id_chain(tmp_path):
    # first "group" of genes starts numbering at 1 (the default)
    a = Annotation()
    a.add(tx((10, 20), sequence="chr1"))
    a.add(tx((30, 40), sequence="chr1"))
    assert a.next_gene_id == 3

    # a following group continues from the previous counter, no overlap
    b = Annotation(start_gene_id=a.next_gene_id)
    b.add(tx((10, 20), sequence="chr2"))
    assert b.next_gene_id == 4

    out = tmp_path / "o.gtf"
    a.to_gtf(out)
    b.to_gtf(out, append=True)
    gene_ids = [ln.split("\t")[8] for ln in out.read_text().splitlines()
                if ln.split("\t")[2] == "gene"]
    assert gene_ids == [
        'gene_id "g1";', 'gene_id "g2";', 'gene_id "g3";',
    ]


# -------------------------------------------------------------------- #
# Writers
# -------------------------------------------------------------------- #
def _sample_annotation():
    ann = Annotation()
    ann.add(Gene(sequence="chr1", strand="+", transcripts=[
        tx((10, 20), (30, 40)),
        tx((10, 25), (30, 40)),
    ]))
    ann.add(tx((100, 130), sequence="chr1", strand="-"))
    return ann


def test_to_gtf_consistent_ids(tmp_path):
    out = tmp_path / "o.gtf"
    _sample_annotation().to_gtf(out)
    lines = out.read_text().splitlines()

    gene_lines = [ln for ln in lines if ln.split("\t")[2] == "gene"]
    assert len(gene_lines) == 2
    assert 'gene_id "g1";' in gene_lines[0]
    assert 'gene_id "g2";' in gene_lines[1]

    # two isoforms of gene g1 share the gene id, differ in transcript id
    # isoforms are numbered from 1 within their gene
    tx_lines = [ln for ln in lines if ln.split("\t")[2] == "transcript"]
    assert 'gene_id "g1"; transcript_id "g1_t1";' in tx_lines[0]
    assert 'gene_id "g1"; transcript_id "g1_t2";' in tx_lines[1]


def test_to_gff3_has_header_and_parents(tmp_path):
    out = tmp_path / "o.gff3"
    _sample_annotation().to_gff3(out)
    text = out.read_text()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert text.splitlines()[0] == "##gff-version 3"
    assert "\tgene\t" in text
    # mRNA points to its gene; children point to the mRNA.
    mrna = [ln for ln in lines if ln.split("\t")[2] == "mRNA"][0]
    assert "ID=g1_t1;Name=g1_t1;Parent=g1" == mrna.split("\t")[8]
    cds = [ln for ln in lines if ln.split("\t")[2] == "CDS"][0]
    assert cds.split("\t")[8] == "ID=g1_t1.CDS.1;Parent=g1_t1"


def test_to_genepred_one_row_per_transcript(tmp_path):
    out = tmp_path / "o.gp"
    _sample_annotation().to_genepred(out)
    rows = out.read_text().splitlines()
    names = [r.split("\t")[0] for r in rows]
    assert names == ["g1_t1", "g1_t2", "g2_t1"]


def test_to_gtf_append(tmp_path):
    out = tmp_path / "o.gtf"
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chr1"))
    ann.to_gtf(out)
    ann.to_gtf(out, append=True)
    genes = [ln for ln in out.read_text().splitlines()
             if ln.split("\t")[2] == "gene"]
    assert len(genes) == 2


def test_set_naming_changes_output(tmp_path):
    ann = Annotation()
    ann.add(tx((10, 20), sequence="chr1"))
    ann.set_naming(
        gene_pattern="GENE{gene_id}",
        transcript_pattern="GENE{gene_id}.iso{transcript_id}",
    )
    out = tmp_path / "o.gtf"
    ann.to_gtf(out)
    text = out.read_text()
    assert 'gene_id "GENE1";' in text
    assert 'transcript_id "GENE1.iso1";' in text


def test_explicit_names_take_precedence(tmp_path):
    ann = Annotation()
    ann.add(Gene(name="MYGENE", sequence="chr1", strand="+", transcripts=[
        tx((10, 20), name="MYTX"),
    ]))
    out = tmp_path / "o.gtf"
    ann.to_gtf(out)
    text = out.read_text()
    assert 'gene_id "MYGENE"; transcript_id "MYTX";' in text


# -------------------------------------------------------------------- #
# GenePred round-trip
# -------------------------------------------------------------------- #
def test_genepred_roundtrip(tmp_path):
    ann = _sample_annotation()
    out = tmp_path / "o.gp"
    ann.to_genepred(out)

    loaded = Annotation.from_genepred(out)
    genes = list(loaded.genes())
    assert [(g.start, g.end, g.strand) for g in genes] == [
        (10, 40, "+"), (10, 40, "+"), (100, 130, "-"),
    ]


# -------------------------------------------------------------------- #
# Fasta-backed sequence extraction
# -------------------------------------------------------------------- #
def test_coding_and_protein_sequence():
    # spliced coding sequence: ATGAAA + TAA = "ATGAAATAA" -> M K *
    fasta = fasta_from_string("ATGAAAGTCTAA")
    seq = fasta[0]
    t = tx((0, 6), (9, 12), sequence=seq.name)
    assert t.coding_sequence(seq).string() == "ATGAAATAA"
    assert t.protein_sequence(seq) == "MK"


def test_coding_sequence_minus_strand():
    fasta = fasta_from_string("ATGAAAGTCTAA")
    seq = fasta[0]
    t = tx((0, 6), sequence=seq.name, strand="-")
    # reverse complement of ATGAAA
    assert t.coding_sequence(seq).string() == "TTTCAT"


def test_coding_sequence_name_mismatch_raises():
    fasta = fasta_from_string("ACGT")
    seq = fasta[0]
    t = tx((0, 4), sequence="other")
    with pytest.raises(KeyError):
        t.coding_sequence(seq)


def test_sequence_to_file_writes_headers(tmp_path):
    fasta = fasta_from_string("ATGAAAGTCTAA")
    seq = fasta[0]
    ann = Annotation()
    ann.add(tx((0, 6), (9, 12), sequence=seq.name))

    out = tmp_path / "prot.fa"
    ann.sequence_to_file("protein", fasta, out, line_width=0)
    lines = out.read_text().splitlines()
    assert lines[0].startswith(">g1|g1_t1|")
    assert lines[1] == "MK"
