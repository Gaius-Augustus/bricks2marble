from typing import Callable, Literal

import numpy as np

from .struct import FASTA, Annotation, FeatureType, GTFEntry, Region, Sequence

HMM_STATE_AGGREGATION = np.array([
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1.],
])


def _split_regions(
    encoded_labels: np.ndarray,
    offset: int = 0,
) -> list[Region]:
    arr = np.array(encoded_labels)
    arr = HMM_STATE_AGGREGATION.argmax(1)[arr]
    arr[(arr > 1)] = 2
    change_points = np.where(np.diff(arr) != 0)[0]
    start_points = np.insert(change_points + 1, 0, 0)
    end_points = np.append(change_points, arr.size - 1)

    features = ["intergenic", "intron", "CDS"]
    regions = [
        Region(
            name=features[arr[start]],
            start=start+offset,
            end=end+offset,
        )
        for start, end in zip(start_points, end_points)
    ]
    return regions


def _transcripts_from_regions(
    regions: list[Region],
) -> tuple[list[Region], list[list[Region]], list[Region]]:
    """Extracts transcript regions (IR to IR) from a given tuples of
    class regions. For each transcript, it extracts the regions of their
    CDS. Additionally, it reports fragmented txs at the start or end of
    the input ranges.

    Args:
        regions (list[Region]): A sequence of regions, classifying parts
            of a genome sequence into different labels.

    Returns:
        initial_tx (list): A fragmented transcript at the start of the
            given regions.
        txs (list of lists): A list of complete transcripts within the
            given regions.
        current_tx (list): A fragmented transcript at the end of the
            given regions.
    """
    initial_tx: list[Region] = []
    txs: list[list[Region]] = []
    current_tx: list[Region] = []

    for region in regions:
        if region.name == 'intergenic':
            if current_tx:
                txs.append(current_tx)
                current_tx = []
        else:
            current_tx.append(region)
    if regions[0].name != 'intergenic' and txs:
        initial_tx = txs[0]
        txs = txs[1:]
    return initial_tx, txs, current_tx


def _merge_reprediction(
    all_tx: list[list[Region]],
    new_tx: list[list[Region]],
    breakpoint: int,
) -> list[list[Region]]:
    """Merges two sets of transcript predictions (`all_tx` and `new_tx`)
    at a specified breakpoint.

    This function integrates predictions from two different prediction
    sets by considering their overlaps and the specified breakpoint. It
    aims to create a combined prediction that respects the continuity of
    transcripts across the breakpoint, favoring the retention of longer
    transcripts or more accurate predictions based on the overlap
    analysis.

    Arguments:
        all_tx (list of tuples): The list of all current transcript
            predictions before the breakpoint. Each element in the list
            is a tuple representing a transcript with its start and end
            positions.
        new_tx (list of tuples): The list of new transcript predictions
            that may overlap with `all_tx` at the breakpoint.
        breakpoint (int): The position in the sequence where the
            division between the old and new predictions is made.

    Returns:
        list[Region]: The merged list of transcript predictions,
            considering the breakpoint and overlaps between `all_tx`
            and `new_tx`.

    The merging process follows these rules:
        - If one of the prediction sets is empty, it returns the
            concatenation of both.
        - If the breakpoint is in the intergenic region (outside the
            range of any transcripts in both sets), it merges the
            predictions without overlapping transcripts.
        - If the breakpoint indicates overlapping regions but no direct
            overlap between transcripts, it concatenates the predictions
            up to and from the breakpoint.
        - If there's an overlap and one of the transcripts surrounding
            the breakpoint is larger, the larger transcript is preferred
            in the merged output.
    """
    overlap1 = 0
    for i, tx in enumerate(all_tx):
        if breakpoint < tx[0].start:
            break
        overlap1 = i
    overlap2 = 0
    for i, tx in enumerate(new_tx):
        if breakpoint < tx[0].start:
            break
        overlap2 = i

    if not all_tx or not new_tx:
        # no tx in one of the predictions
        return all_tx + new_tx
    elif (breakpoint > all_tx[overlap1][-1].end
            and breakpoint > new_tx[overlap2][-1].end):
        # breakpoint already in intergenic region of both sets
        return all_tx[:overlap1+1] + new_tx[overlap2+1:]
    elif all_tx[overlap1][-1].end < new_tx[overlap2][0].start:
        # breakpoint in one of the transcripts but they don't overlap
        return all_tx[:overlap1+1] + new_tx[overlap2:]
    elif (all_tx[overlap1][-1].end - all_tx[overlap1][0].start
            > new_tx[overlap2][-1].end - new_tx[overlap2][0].start):
        # tx from all_tx is larger so keep it instead of the one from new_tx
        return all_tx[:overlap1+1] + new_tx[overlap2+1:]
    else:
        # tx from new_tx is larger so keep it instead of the one from all_tx
        return all_tx[:overlap1] + new_tx[overlap2:]


def GTF_from_model(
    fasta: FASTA,
    predict_func: Callable[[FASTA], np.ndarray],
    model_name: str = "Model",
    strand: Literal["+", "-"] = "+",
    tx_id: int = 0,
    filter_transcripts: bool = True,
) -> Annotation:
    """Generate a genome annotation using a nucleotide sequence and a
    function that outputs feature labels of regions of interest.

    Args:
        fasta (FASTA): A :class:`FASTA` object containing the nucleotide
            sequences of interest.
        predict_func (Callable): A function that takes a :class:`FASTA`
            object as input and outputs a numpy array of shape ``(B, T,
            D)``.
        model_name (str): Name of the model that is used, or any other
            identifier. This will only be listed as the 'source' in the
            GTF file.
        strand ("+" or "-"): Whether the annotation is for the forward
            or backward strand. Defaults to "+".
        tx_id (int): ? # TODO
        filter_transcripts (bool): ? # TODO
    """
    annotation = Annotation()
    ranges: dict[str, list[list[Region]]] = {}

    labels = predict_func(fasta)

    repred_seqs = []
    repred_index = []

    for i in range(labels.shape[0]-1):
        if (fasta.segments[i].name == fasta.segments[i+1].name
                and labels[i, -1] != labels[i+1, 0]):
            repred_seqs.append(Sequence(
                np.concatenate(
                    (fasta.nuc[i], fasta.nuc[i+1]),
                    axis=0,
                ),
                name=fasta.segments[i].name,
                start=fasta.segments[i].start,
                end=fasta.segments[i+1].end,
            ))
            repred_index.append(i)

    if len(repred_seqs) > 0:
        repred_out = predict_func(FASTA(repred_seqs))

    re_txs = None
    end_fragment = []

    for i, (y, c) in enumerate(zip(labels, fasta.segments)):
        regions = _split_regions(y, c.start)
        is_ir = 'intergenic' in [r.name for r in regions]
        coord_diff = 0 if i == 0 else (c.end - fasta.segments[i-1].start)
        start_fragment, txs, new_end_fragment = _transcripts_from_regions(
            regions
        )

        if c.name not in ranges:
            ranges[c.name] = []

        # if the start of the first fragmented tx matches the fragment
        # from the last chunk, combine them
        if (not re_txs and is_ir and end_fragment and start_fragment
                and labels[i-1, -1] == labels[i, 0]):
            end_fragment[-1].start = start_fragment[0].start
            end_fragment += start_fragment[1:]
            ranges[c.name] += [end_fragment]

        if is_ir and txs:
            if re_txs:
                ranges[c.name] = _merge_reprediction(
                    ranges[c.name],
                    txs,
                    c.start + coord_diff//2,
                )
            else:
                ranges[c.name] += txs
        if is_ir:
            end_fragment = new_end_fragment

        re_txs = None
        if repred_index and i == repred_index[0]:
            repred_index.pop(0)
            c_re = Region(
                name=c.name,
                start=c.start,
                end=c.end+coord_diff,
                strand=strand,  # TODO unsure
            )
            current_re = repred_out[0]
            repred_out = repred_out[1:]
            if c_re.strand == '-':
                current_re = current_re[::-1]
            re_ranges = _split_regions(current_re, c_re.start)
            start_fragment, re_txs, new_end_fragment = (
                _transcripts_from_regions(re_ranges)
            )

            if (not is_ir and end_fragment and start_fragment \
                    and labels[i-1,-1] == current_re[0]):
                end_fragment[-1].end = start_fragment[0].end
                end_fragment += start_fragment[1:]
                ranges[c.name] += [end_fragment]
            if re_txs:
                ranges[c.name] = _merge_reprediction(
                    ranges[c.name],
                    re_txs,
                    c_re.end + coord_diff//2,
                )

            end_fragment = new_end_fragment

    for seq in ranges:
        phase = -1
        for tx in ranges[seq]:
            tx_id += 1
            t_id = f'g{tx_id}.t1'
            g_id = f'g{tx_id}'
            phase = 0
            for r in tx:
                annotation.add(
                    GTFEntry(
                        name=seq,
                        source=model_name,
                        feature=FeatureType(r.name),
                        start=r.start,
                        end=r.end,
                        score=None,
                        strand=strand,
                        frame=phase,  # type: ignore
                        attributes=f"gene_id \"{g_id}\"; "
                                   f"transcript_id \"{t_id}\";",
                    ),
                    gene_id=g_id,
                    transcript_id=t_id,
                )
                if r.name == 'CDS':
                    phase = (3 - (r.end - r.start + 1 - phase) % 3) % 3

    # remove_tx = []
    # for gene in annotation.genes.values():
    #     for tx in gene.transcripts.values():
    #         tx.check_splits()
    #         if filter_transcripts and tx.get_cds_len() < 201:
    #             remove_tx.append((gene.id, tx.id))
    #         else:
    #             tx.redo_phase()

    # for g_id, t_id in remove_tx:
    #     annotation.genes[g_id].transcripts.pop(t_id)

    annotation.finalize()
    return annotation
