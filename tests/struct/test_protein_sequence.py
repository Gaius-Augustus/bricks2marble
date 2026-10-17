from bricks2marble.io import fasta_from_string
from bricks2marble.struct import CDS, Transcript


def _tx(seq: str, strand: str = "+") -> tuple[Transcript, object]:
    fasta = fasta_from_string(seq)
    tx = Transcript(sequence="seq1", strand=strand, cds=[CDS(start=0, end=len(seq))])
    return tx, fasta["seq1"]


def test_initiator_methionine_for_alternative_start():
    tx, seq = _tx("GTGGCCTAA")
    assert tx.protein_sequence(seq, translation_table=11) == "VA"
    assert tx.protein_sequence(
        seq, translation_table=11, initiator_methionine=True,
    ) == "MA"


def test_initiator_methionine_leaves_non_start_codons():
    # GCC is not a start codon in table 11, so nothing changes
    tx, seq = _tx("GCCGCCTAA")
    assert tx.protein_sequence(
        seq, translation_table=11, initiator_methionine=True,
    ) == "AA"
    # ATG is a methionine anyway
    tx, seq = _tx("ATGGCCTAA")
    assert tx.protein_sequence(
        seq, translation_table=1, initiator_methionine=True,
    ) == "MA"


def test_initiator_methionine_follows_the_table():
    # GTG is no start codon in the standard code
    tx, seq = _tx("GTGGCCTAA")
    assert tx.protein_sequence(
        seq, translation_table=1, initiator_methionine=True,
    ) == "VA"
