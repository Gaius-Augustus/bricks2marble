from ..struct import FASTA, Annotation, FeatureType


def check_annotation_boundaries(
    annotation: "Annotation",
    fasta: "FASTA",
    start_codons: list[str] | bool = True,
    stop_codons: list[str] | bool = True,
    intron_begin: list[str] | bool = True,
    intron_end: list[str] | bool = True,
    remove: bool = False,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Checks the given annotation for having start and stop codons at
    all transcript borders or restrictive patterns for intron begins and
    ends. Also checks for any transcripts that might be out-of-bounds
    for the given fasta.

    Args:
        annotation (Annotation): The annotation to verify.
        fasta (FASTA): The corresponding fasta object to the annotation.
        start_codons (list[str], optional): A list of strings of
            possible start codons or a boolean value. If true, defaults
            to only "ATG" and if false, does no checks for start codons.
        stop_codons (list[str], optional): A list of strings of
            possible stop codons or a boolean value. If true, defaults
            to "TAG", "TAA" or "TGA" and if false, does no checks for
            stop codons.
        intron_begin (list[str], optional): A list of strings of
            possible begin patterns of introns, or a boolean value. If
            true, defaults to only "GT" and if false, does no checks for
            begin patterns.
        intron_end (list[str], optional): A list of strings of
            possible end patterns of introns, or a boolean value. If
            true, defaults to only "AG" and if false, does no checks for
            end patterns.
        remove (bool, optional): If set to true, removes all bad
            transcripts from the annotation.

    Returns:
        (tuple of lists): Five lists (gene id, transcript id, start,
        end, strand) for:
            - positions with missing start codons
            - positions with missing stop codons
            - positions with missing intron begin patterns
            - positions with missing intron end patterns
            - out-of-bounds positions
    """
    if isinstance(start_codons, bool) and start_codons: start_codons = ["ATG"]
    if isinstance(stop_codons, bool) and stop_codons:
        stop_codons = ["TAG", "TAA", "TGA"]
    if isinstance(intron_begin, bool) and intron_begin: intron_begin = ["GT"]
    if isinstance(intron_end, bool) and intron_end: intron_end = ["AG"]

    reverse = str.maketrans("ACGTNacgt", "TGCANtgca")

    wrong_start = []
    wrong_stop = []
    wrong_begin = []
    wrong_end = []
    out_of_range = []
    seqname = next(iter(next(iter(annotation)))).seqname
    seq = fasta[seqname].string()
    for gene in annotation:
        for tx in gene:
            if tx.seqname != seqname:
                seqname = tx.seqname
                seq = fasta[seqname].string()

            if tx.end > len(seq):
                out_of_range.append(
                    (gene.id, tx.id, tx.start, tx.end, tx.strand)
                )
                continue

            if start_codons:
                kmer = (
                    seq[tx.start:tx.start+3] if tx.strand == "+"
                    else seq[tx.end-3:tx.end][::-1].translate(reverse)
                )
                if kmer.upper() not in start_codons:
                    wrong_start.append(
                        (gene.id, tx.id, tx.start, tx.end, tx.strand)
                    )

            if stop_codons:
                kmer = (
                    seq[tx.end-3:tx.end] if tx.strand == "+"
                    else seq[tx.start:tx.start+3][::-1].translate(reverse)
                )
                if kmer.upper() not in stop_codons:
                    wrong_stop.append(
                        (gene.id, tx.id, tx.start, tx.end, tx.strand)
                    )

            for entry in tx.entries:
                if entry.feature == FeatureType.Intron:
                    if intron_begin:
                        kmer = (
                            seq[entry.start:entry.start+2] if tx.strand == "+"
                            else
                            seq[entry.end-2:entry.end][::-1].translate(reverse)
                        )
                        if kmer.upper() not in intron_begin:
                            wrong_begin.append((gene.id, tx.id, entry.start,
                                                entry.end, tx.strand))
                    if intron_end:
                        kmer = (
                            seq[entry.end-2:entry.end] if tx.strand == "+"
                            else
                            seq[entry.start:entry.start+2][::-1].translate(
                                reverse
                            )
                        )
                        if kmer.upper() not in intron_end:
                            wrong_end.append((gene.id, tx.id, entry.start,
                                              entry.end, tx.strand))

    if remove:
        for gid, tid, _, _, _ in out_of_range:
            annotation[gid]._transcripts.pop(tid)
        for gid, tid, _, _, _ in wrong_start:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue
        for gid, tid, _, _, _ in wrong_stop:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue
        for gid, tid, _, _, _ in wrong_begin:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue
        for gid, tid, _, _, _ in wrong_end:
            try:
                annotation[gid]._transcripts.pop(tid)
            except KeyError:
                continue

    return wrong_start, wrong_stop, wrong_begin, wrong_end, out_of_range
