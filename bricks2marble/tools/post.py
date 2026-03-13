import numpy as np

from ..io import fasta_from_string
from ..struct import Annotation, Fasta, FeatureType, Transcript
from ..struct.fasta import complement, nucleotides_to_kmers


def check_min_coding_length(
    annotation: Annotation,
    length: int,
    remove: bool = False,
) -> list[Transcript]:
    """Returns transcripts with a shorter coding region than the given
    length.

    Args:
        annotation (Annotation): The annotation to verify.
        length (int, optional): Minimal length of coding regions.
        remove (bool, optional): If set to true, removes all found
            transcripts from the annotation.
    """
    txs = []
    for gene in annotation:
        for tx in gene:
            if tx.cds_length() < length: txs.append(tx)

    if remove:
        for tx in txs: annotation.remove(tx)
    return txs


def check_inframe_stop_codons(
    annotation: Annotation,
    fasta: Fasta,
    codons: set[str] | None = None,
    remove: bool = False,
) -> list[Transcript]:
    """Checks the annotation for inframe stop codons.

    Args:
        annotation (Annotation): The annotation to verify.
        fasta (Fasta): The corresponding fasta object to the annotation.
        codons (list[str], optional): A list of strings of
            possible stop codons or a boolean value. Defaults to "TAG",
            "TAA" or "TGA".
        remove (bool, optional): If set to true, removes all found
            transcripts from the annotation.

    Returns:
        (list of Transcript): A list of transcripts that contain inframe
            stop codons.
    """
    if codons is None: codons = {"TAG", "TAA", "TGA"}

    c = [nucleotides_to_kmers(fasta_from_string(i)[0].flat) for i in codons]

    txs = []
    for sequence in fasta:
        seq = sequence.flat
        for gene in annotation.in_sequence(sequence.name):
            for tx in gene:
                if tx.cds_length() < 6: continue
                s = np.r_[
                    *(seq[cds[0]:cds[1]] for cds in tx.coords(FeatureType.CDS))
                ]
                if tx.strand == "-": s = complement(s, reverse=True)
                s3 = nucleotides_to_kmers(s)[:-1]

                if np.any(np.isin(s3, c)): txs.append(tx)

    if remove:
        for tx in txs: annotation.remove(tx)
    return txs


def check_exon_boundaries(
    annotation: Annotation,
    fasta: Fasta,
    start_codons: set[str] | None = None,
    stop_codons: set[str] | None = None,
    intron_begin: set[str] | None = None,
    intron_end: set[str] | None = None,
    remove: bool = False,
) -> tuple[
    list[Transcript], list[Transcript], list[Transcript], list[Transcript]
]:
    """Checks the given annotation for having start and stop codons at
    all transcript borders or restrictive patterns for intron begins and
    ends.

    Lists of transcripts that do not meet the criteria are returned and
    can optionally be removed from the annotation. The lists are
    exclusive. If a transcript has more than one type of error, it only
    appears in one list.

    Args:
        annotation (Annotation): The annotation to verify.
        fasta (Fasta): The corresponding fasta object to the annotation.
        start_codons (set[str], optional): A set of strings of possible
            start codons. Defaults to {"ATG"}.
        stop_codons (set[str], optional): A set of strings of
            possible stop codons. Defaults to {"TAG", "TAA" or "TGA"}.
        intron_begin (set[str], optional): A set of strings of possible
            begin patterns of introns. Defaults to {"GT"}.
        intron_end (set[str], optional): A set of strings of possible
            end patterns of introns. Defaults to {"AG"}.
        remove (bool, optional): If set to true, removes all bad
            transcripts from the annotation.

    Returns:
        (tuple of lists): Four lists of transcripts with:
            - wrong start codons
            - wrong stop codons
            - wrong intron begin patterns
            - wrong intron end patterns
    """
    if start_codons is None: start_codons = {"ATG"}
    if stop_codons is None: stop_codons = {"TAG", "TAA", "TGA"}
    if intron_begin is None: intron_begin = {"GT"}
    if intron_end is None: intron_end = {"AG"}

    wrong_start = []
    wrong_stop = []
    wrong_begin = []
    wrong_end = []

    for gene in annotation:
        seq = fasta[gene.seqname]
        for tx in gene:

            kmer = (
                seq.positions(tx.start, tx.start+3) if tx.strand == "+"
                else seq.positions(tx.end-3, tx.end).complement(reverse=True)
            ).string()
            if kmer.upper() not in start_codons:
                wrong_start.append(tx)
                continue

            kmer = (
                seq.positions(tx.end-3, tx.end) if tx.strand == "+" else
                seq.positions(tx.start, tx.start+3).complement(reverse=True)
            ).string()
            if kmer.upper() not in stop_codons:
                wrong_stop.append(tx)
                continue

            for entry in tx.entries:
                if entry.feature == FeatureType.Intron:
                    kmer = (
                        seq.positions(entry.start, entry.start+2)
                        if tx.strand == "+" else
                        seq.positions(
                            entry.end-2, entry.end,
                        ).complement(reverse=True)
                    ).string()
                    if kmer.upper() not in intron_begin:
                        wrong_begin.append(tx)
                        continue
                    kmer = (
                        seq.positions(entry.end-2, entry.end)
                        if tx.strand == "+" else
                        seq.positions(
                            entry.start, entry.start+2,
                        ).complement(reverse=True)
                    ).string()
                    if kmer.upper() not in intron_end:
                        wrong_end.append(tx)
                        continue

    if remove:
        for tx in (wrong_start + wrong_stop + wrong_begin + wrong_end):
            annotation.remove(tx)

    return wrong_start, wrong_stop, wrong_begin, wrong_end


def check_coding_repeats(
    annotation: Annotation,
    fasta: Fasta,
    remove: bool = False,
) -> list[Transcript]:
    repeats = []
    for gene in annotation:
        seq = fasta[gene.seqname]
        for tx in gene:
            if tx.coding_sequence(seq).is_repeat_masked():
                repeats.append(tx)
    if remove:
        for tx in repeats: annotation.remove(tx)
    return repeats


def check_out_of_bounds(
    annotation: Annotation,
    fasta: Fasta,
    remove: bool = False,
) -> list[Transcript]:
    """Checks the given annotation for any transcripts that are
    out-of-bounds for the given fasta.

    Args:
        annotation (Annotation): The annotation to verify.
        fasta (Fasta): The corresponding fasta object to the annotation.
        remove (bool, optional): If set to true, removes all bad
            transcripts from the annotation.

    Returns:
        (list of Transcript): A list of out-of-bounds transcripts.
    """
    txs = []
    for gene in annotation:
        for tx in gene:
            if tx.end > fasta[tx.seqname].size:
                txs.append(tx)
    if remove:
        for tx in txs: annotation.remove(tx)
    return txs
