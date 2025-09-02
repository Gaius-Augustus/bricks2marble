from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..struct import FASTA, Annotation


def check_annotation_boundaries(
    annotation: "Annotation",
    fasta: "FASTA",
    start_codons: list[str] | None = None,
    stop_codons: list[str] | None = None,
    remove: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Checks the given annotation for having start and stop codons at
    all transcript borders. Also checks for any transcripts that might
    be out-of-bounds for the given fasta.

    Args:
        annotation (Annotation): The annotation to verify.
        fasta (FASTA): The corresponding fasta object to the annotation.
        start_codons (list[str], optional): A list of strings of
            possible start codons. Defaults to only "ATG".
        stop_codons (list[str], optional): A list of strings of
            possible stop codons. Defaults to "TAG", "TAA" or "TGA".
        remove (bool, optional): If set to true, removes all bad
            transcripts from the annotation.

    Returns:
        (tuple of lists): Three lists of gene ids and transcript ids
        for:
            - positions with missing start codons
            - positions with missing stop codons
            - out-of-bounds positions
    """
    if start_codons is None: start_codons = ["ATG"]
    if stop_codons is None: stop_codons = ["TAG", "TAA", "TGA"]

    reverse = str.maketrans("ACGTNacgt", "TGCANtgca")

    wrong_start = []
    wrong_stop = []
    out_of_range = []
    seqname = next(iter(next(iter(annotation)))).seqname
    seq = fasta[seqname].string()
    for gene in annotation:
        for tx in gene:
            if tx.seqname != seqname:
                seqname = tx.seqname
                seq = fasta[seqname].string()

            if tx.end > len(seq):
                out_of_range.append((gene.id, tx.id))
                continue

            kmer = (
                seq[tx.start:tx.start+3] if tx.strand == "+"
                else seq[tx.end-3:tx.end][::-1].translate(reverse)
            )
            if kmer.upper() not in start_codons:
                wrong_start.append((gene.id, tx.id))

            kmer = (
                seq[tx.end-3:tx.end] if tx.strand == "+"
                else seq[tx.start:tx.start+3][::-1].translate(reverse)
            )
            if kmer.upper() not in stop_codons:
                wrong_stop.append((gene.id, tx.id))

    if remove:
        for gid, tid in out_of_range:
            annotation[gid]._transcripts.pop(tid)
        for gid, tid in wrong_start:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue
        for gid, tid in wrong_stop:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue

    return wrong_start, wrong_stop, out_of_range
