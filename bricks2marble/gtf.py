from timeit import default_timer
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
    strand: Literal["+", "-"] = "+",
) -> list[Region]:
    """Splits a sequence of HMM states into a sequence of regions
    "intergenic", "intron" or "CDS".
    """
    arr = np.array(encoded_labels)
    arr = HMM_STATE_AGGREGATION.argmax(1)[arr]
    arr[(arr > 1)] = 2
    change_points = np.where(np.diff(arr) != 0)[0]
    start_points = np.insert(change_points + 1, 0, 0)
    end_points = np.append(change_points + 1, arr.size)

    features = ["intergenic", "intron", "CDS"]
    regions = [
        Region(
            name=features[arr[start]],
            start=start+offset,
            end=end+offset,
            strand=strand,
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


def _annotation_from_dict(
    entries_fwd: dict[str, list[list[Region]]],
    entries_bwd: dict[str, list[list[Region]]],
    model_name: str = "Model",
) -> Annotation:
    annotation = Annotation()
    tx_id = 0
    for seq in entries_fwd:
        phase = -1
        len_fwd = 0 if seq not in entries_fwd else len(entries_fwd[seq])
        len_bwd = 0 if seq not in entries_bwd else len(entries_bwd[seq])
        while len_fwd + len_bwd > 0:
            fwd = False
            if len_fwd and len_bwd > 0 and (
                entries_fwd[seq][0][0].start >= entries_bwd[seq][0][0].start
            ):
                fwd = True
            elif len_fwd > 0:
                fwd = True

            if fwd:
                tx = entries_fwd[seq].pop(0)
            else:
                tx = entries_bwd[seq].pop(0)

            tx_id += 1
            t_id = f"g{tx_id}.t1"
            g_id = f"g{tx_id}"
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
                        strand="+" if fwd else "-",
                        frame=phase,  # type: ignore
                        attributes=f"gene_id \"{g_id}\"; "
                                   f"transcript_id \"{t_id}\";",
                    ),
                    gene_id=g_id,
                    strand="+" if fwd else "-",
                    transcript_id=t_id,
                )
                if r.name == "CDS":
                    phase = (3 - (r.end - r.start - phase) % 3) % 3
            len_fwd = 0 if seq not in entries_fwd else len(entries_fwd[seq])
            len_bwd = 0 if seq not in entries_bwd else len(entries_bwd[seq])
    return annotation


def GTF_from_model(
    fasta: FASTA,
    predict_func: Callable[[FASTA],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ],
    model_name: str = "Model",
    verbose: bool = True,
) -> Annotation:
    """Generate a genome annotation using a nucleotide sequence and a
    function that outputs feature labels of regions of interest.

    Args:
        fasta (FASTA): A :class:`FASTA` object containing the nucleotide
            sequences of interest.
        predict_func (Callable): A function that takes a :class:`FASTA`
            object as input and outputs one or two numpy arrays. Each of
            these array has to have shape ``(N, T)``, where ``N`` is
            the number of sequences in the given `fasta` and ``T`` is
            its chunk size. The output should be integers of states from
            the :class:`bricks2marble.tf.HMM`. The first numpy array is
            a prediction on the forward strand, the second a prediction
            on the reverse strand. One of them can be missing.
        model_name (str): Name of the model that is used, or any other
            identifier. This will only be listed as the 'source' in the
            GTF file.
    """
    if verbose: start_time = default_timer()

    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Start initial prediction of "
        f"{fasta.N} sequences.",
        flush=True,
    )

    labels_fwd, labels_bwd = predict_func(fasta)

    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Searching for errors.",
        flush=True,
    )

    repred_seqs = []
    repred_index = []
    repred_strand = []
    for i in range(fasta.N-1):
        if (fasta.segments[i].name == fasta.segments[i+1].name):
            fwd = bwd = False

            if labels_fwd is not None and not (
                labels_fwd[i, -1] == labels_fwd[i+1, 0] == 0
            ):
                fwd = True
            if labels_bwd is not None and not(
                labels_bwd[i, -1] == labels_bwd[i+1, 0] == 0
            ):
                bwd = True

            if fwd or bwd:
                repred_seqs.append(Sequence(
                    np.expand_dims(
                        np.concatenate((fasta.nuc[i], fasta.nuc[i+1]), axis=0),
                        axis=0,
                    ),
                    name=fasta.segments[i].name,
                    start=fasta.segments[i].start,
                    end=fasta.segments[i+1].end,
                ))
                repred_index.append(i)
                if fwd and bwd:
                    repred_strand.append(2)
                else:
                    repred_strand.append(0 if fwd else 1)

    if len(repred_seqs) > 0:
        if verbose: print(
            f"[{default_timer()-start_time:.4f}s] Mismatches: "
            f"{repred_strand.count(0)} (+) | {repred_strand.count(1)} (-) | "
            f"{repred_strand.count(2)} (+/-). "
            f"Repredicting {len(repred_seqs)} sequences.",
            flush=True,
        )
        repred_fwd, repred_bwd = predict_func(FASTA(repred_seqs))

    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Forming regions.",
        flush=True,
    )

    entries_fwd: dict[str, list[list[Region]]] = {}
    entries_bwd: dict[str, list[list[Region]]] = {}
    re_txs_f = None
    re_txs_b = None
    last_end_f = []
    last_end_b = []
    for i, segment in enumerate(fasta.segments):

        coord_diff = 0 if i == 0 else (segment.end - fasta.segments[i-1].start)

        if labels_fwd is not None:
            regions_fwd = _split_regions(
                labels_fwd[i],
                segment.start,
                strand="+",
            )
            start_f, txs_f, end_f = _transcripts_from_regions(regions_fwd)
            if segment.name not in entries_fwd: entries_fwd[segment.name] = []

            is_ir_f = 'intergenic' in [r.name for r in regions_fwd]
            if (re_txs_f is None and is_ir_f and start_f and last_end_f and
                (i == 0 or labels_fwd[i-1, -1] == labels_fwd[i, 0])
            ):
                last_end_f[-1].start = start_f[0].start
                last_end_f += start_f[1:]
                entries_fwd[segment.name] += [last_end_f]

            if is_ir_f and len(txs_f) > 0:
                if re_txs_f is not None:
                    entries_fwd[segment.name] = _merge_reprediction(
                        entries_fwd[segment.name],
                        txs_f,
                        segment.start + coord_diff//2,
                    )
                else:
                    if i == 0 and len(start_f) > 0:
                        entries_fwd[segment.name] += [start_f]
                    entries_fwd[segment.name] += txs_f
            if is_ir_f:
                last_end_f = end_f

        if labels_bwd is not None:
            regions_bwd = _split_regions(
                labels_bwd[i],
                segment.start,
                strand="-",
            )
            start_b, txs_b, end_b = _transcripts_from_regions(regions_bwd)
            if segment.name not in entries_bwd: entries_bwd[segment.name] = []

            is_ir_b = 'intergenic' in [r.name for r in regions_bwd]
            if (re_txs_b is None and is_ir_b and start_b and last_end_b and
                (i == 0 or labels_bwd[i-1, -1] == labels_bwd[i, 0])
            ):
                last_end_b[-1].start = start_b[0].start
                last_end_b += start_b[1:]
                entries_bwd[segment.name] += [last_end_b]

            if is_ir_b and len(txs_b) > 0:
                if re_txs_b is not None:
                    entries_bwd[segment.name] = _merge_reprediction(
                        entries_bwd[segment.name],
                        txs_b,
                        segment.start + coord_diff//2,
                    )
                else:
                    if i == 0 and len(start_b) > 0:
                        entries_bwd[segment.name] += [start_b]
                    entries_bwd[segment.name] += txs_b
            if is_ir_b:
                last_end_b = end_b

        re_txs_f = None
        re_txs_b = None
        if len(repred_index) > 0 and i == repred_index[0]:
            repred_index.pop(0)
            strand = repred_strand.pop(0)

            if strand == 0 or strand == 2:
                c_re = Region(
                    name=segment.name,
                    start=segment.start,
                    end=segment.end+coord_diff,
                    strand="+",
                )
                current_re = repred_fwd[0]  # type: ignore
                repred_fwd = repred_fwd[1:]  # type: ignore
                re_ranges = _split_regions(current_re, c_re.start)
                start_f, re_txs, end_f = _transcripts_from_regions(re_ranges)

                if (
                    not is_ir_f
                    and last_end_f
                    and start_f
                    and labels_fwd[i-1, -1] == current_re[0]  # type: ignore
                ):
                    last_end_f[-1].end = start_f[0].end
                    last_end_f += start_f[1:]
                    entries_fwd[segment.name] += [last_end_f]
                if re_txs:
                    entries_fwd[segment.name] = _merge_reprediction(
                        entries_fwd[segment.name],
                        re_txs,
                        c_re.end + coord_diff//2,
                    )
                last_end_f = end_f

            if strand == 1 or strand == 2:
                c_re = Region(
                    name=segment.name,
                    start=segment.start,
                    end=segment.end+coord_diff,
                    strand="-",
                )
                current_re = repred_bwd[0]  # type: ignore
                repred_bwd = repred_bwd[1:]  # type: ignore
                re_ranges = _split_regions(current_re, c_re.start)
                start_b, re_txs, end_b = _transcripts_from_regions(re_ranges)

                if (
                    not is_ir_b
                    and last_end_b
                    and start_b
                    and labels_bwd[i-1, -1] == current_re[0]  # type: ignore
                ):
                    last_end_b[-1].end = start_b[0].end
                    last_end_b += start_b[1:]
                    entries_bwd[segment.name] += [last_end_b]
                if re_txs:
                    entries_bwd[segment.name] = _merge_reprediction(
                        entries_bwd[segment.name],
                        re_txs,
                        c_re.end + coord_diff//2,
                    )
                last_end_b = end_b

    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Creating GTF entries.",
        flush=True,
    )

    annotation = _annotation_from_dict(
        entries_fwd,
        entries_bwd,
        model_name=model_name,
    )

    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Finalizing.",
        flush=True,
    )
    annotation.finalize()
    if verbose: print(
        f"[{default_timer()-start_time:.4f}s] Done.",
        flush=True,
    )
    return annotation
