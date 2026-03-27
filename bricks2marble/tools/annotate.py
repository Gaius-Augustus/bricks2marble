import gzip
import shutil
import tempfile
import textwrap
from collections.abc import Container
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from ..io import iterate_sequences
from ..log import log_it, setup_logging
from ..struct import CDS, Annotation, Fasta, Region, Sequence, Transcript
from .types import allowed, convert

HMM_STATE_AGGREGATION = np.array([
    [1., 0., 0., 0., 0.],  # IR
    [0., 1., 0., 0., 0.],  # I0
    [0., 1., 0., 0., 0.],  # I1
    [0., 1., 0., 0., 0.],  # I2
    [0., 0., 1., 0., 0.],  # E0
    [0., 0., 0., 1., 0.],  # E1
    [0., 0., 0., 0., 1.],  # E2
    [0., 0., 1., 0., 0.],  # Start
    [0., 0., 0., 1., 0.],  # EI0
    [0., 0., 0., 0., 1.],  # EI1
    [0., 0., 1., 0., 0.],  # EI2
    [0., 0., 0., 0., 1.],  # IE0
    [0., 0., 1., 0., 0.],  # IE1
    [0., 0., 0., 1., 0.],  # IE2
    [0., 0., 0., 0., 1.],  # Stop
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
) -> list[list[Region]]:
    """Extracts transcript regions (IR to IR) from a given tuples of
    class regions. For each transcript, it extracts the regions of their
    CDS. Only complete transcripts that have an intergenic region before
    and after them are considered.

    Args:
        regions (list[Region]): A sequence of regions, classifying parts
            of a genome sequence into different labels.
    """
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
        txs = txs[1:]
    return txs


def _annotation_from_dict(
    entries_fwd: dict[str, list[list[Region]]],
    entries_bwd: dict[str, list[list[Region]]],
    first_tx_id: int = 0,
) -> tuple[Annotation, int]:
    annotation = Annotation()
    tx_id = first_tx_id
    for seq in sorted(set(entries_fwd) | set(entries_bwd)):
        len_fwd = 0 if seq not in entries_fwd else len(entries_fwd[seq])
        len_bwd = 0 if seq not in entries_bwd else len(entries_bwd[seq])
        while len_fwd + len_bwd > 0:
            fwd = (len_fwd > 0 and len_bwd == 0) or (
                len_fwd > 0 and len_bwd > 0 and
                entries_fwd[seq][0][0].start <= entries_bwd[seq][0][0].start
            )
            if fwd:
                tx = entries_fwd[seq].pop(0)
            else:
                tx = entries_bwd[seq].pop(0)

            tx_id += 1
            t_id = f"g{tx_id}.t1"
            # g_id = f"g{tx_id}"
            annotation.add(Transcript(
                name=t_id,
                sequence=seq,
                strand="+" if fwd else "-",
                cds=[
                    CDS(start=r.start, end=r.end)
                    for r in tx if r.name == "CDS"
                ],
            ))
            len_fwd = 0 if seq not in entries_fwd else len(entries_fwd[seq])
            len_bwd = 0 if seq not in entries_bwd else len(entries_bwd[seq])
    return annotation, tx_id


def _find_mismatches(
    pred: np.ndarray,
    exon_at_boundary: int | None = None,
) -> np.ndarray:
    mask = np.logical_or(
        pred[:-1, -1] != pred[1:, 0],
        ~np.isin(pred[:-1, -1], [0, 1, 2, 3]),
    )
    if exon_at_boundary is not None:
        mask = np.logical_or(
            mask,
            np.any(np.isin(pred[:-1, -exon_at_boundary:], [4, 5, 6]), axis=-1),
        )
        mask = np.logical_or(
            mask,
            np.any(np.isin(pred[1:, :exon_at_boundary], [4, 5, 6]), axis=-1),
        )
    return np.where(mask)[0]


def _merge_replace_center(
    left: np.ndarray,
    right: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    repred_t = center.size // 2
    left_match = np.where((left[-repred_t:] - center[:repred_t])[::-1] == 0)[0]
    right_match = np.where((center[repred_t:] - right[:repred_t]) == 0)[0]
    if left_match.size == 0 or right_match.size == 0:
        return left, right, False
    if left_match.size > 0 and (l := left_match[-1]) > 0:
        left[-l:] = center[repred_t-l:repred_t]
    if right_match.size > 0 and (r := right_match[-1]) > 0:
        right[:r] = center[repred_t:repred_t+r]
    return left, right, True


def _first_non_zero(x: np.ndarray) -> int:
    """Source:
    https://stackoverflow.com/questions/7632963/
        numpy-find-first-index-of-value-fast
    """
    if x.size == 0: return -1
    idx = x.view(bool).argmax() // x.itemsize
    idx = idx if x[idx] else -1
    return idx


def _annotate(
    fasta: Fasta,
    predict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ],
    repredict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ] | None = None,
    reprediction_factor: float = 0.5,
    exon_at_boundary: int | None = None,
    first_tx_id: int = 0,
    concat_strand_to_reprediction: bool = False,
) -> tuple[Annotation, int]:
    log_it(f"Start initial prediction of {fasta.N} chunks.")
    labels_fwd, labels_bwd = predict_func(fasta)
    log_it("Searching for errors.")

    repred_seqs = []
    repred_index = []
    repred_strand = []
    repred_sequence = []
    repred_t = int(fasta.T * reprediction_factor)
    total_mis_fwd = 0
    total_mis_bwd = 0
    total_mis_both = 0

    shift = 0
    for seq_i, seq in enumerate(fasta):
        if labels_fwd is not None:
            mis_fwd = _find_mismatches(
                labels_fwd[shift:shift+seq.N, :],
                exon_at_boundary=exon_at_boundary,
            ) + shift
        else:
            mis_fwd = np.array([])
        if labels_bwd is not None:
            mis_bwd = _find_mismatches(
                labels_bwd[shift:shift+seq.N, :],
                exon_at_boundary=exon_at_boundary,
            ) + shift
        else:
            mis_bwd = np.array([])
        dupli = np.isin(mis_fwd, mis_bwd)
        mis_both = mis_fwd[dupli]
        mis_bwd = mis_bwd[~np.isin(mis_bwd, mis_fwd)]
        mis_fwd = mis_fwd[~dupli]
        mis = np.r_[mis_fwd, mis_bwd, mis_both]
        mis = mis.astype(int)
        total_mis_fwd += len(mis_fwd)
        total_mis_bwd += len(mis_bwd)
        total_mis_both += len(mis_both)
        repred_index.extend(mis)
        repred_sequence.extend([seq_i]*len(mis))
        repred_strand.extend(np.r_[
            np.zeros(len(mis_fwd), dtype=np.int32),
            np.ones(len(mis_bwd), dtype=np.int32),
            np.ones(len(mis_both), dtype=np.int32)+1,
        ])
        if mis.size > 0:
            cseq = Sequence(np.concatenate(
                (fasta.nuc[mis, -repred_t:], fasta.nuc[mis+1, :repred_t]),
                axis=-1,
            ), name=seq.name)
            if concat_strand_to_reprediction:
                strands = np.r_[
                    np.zeros((len(mis_fwd), 2*repred_t), dtype=np.uint8),
                    np.ones((len(mis_bwd), 2*repred_t), dtype=np.uint8),
                    np.ones((len(mis_both), 2*repred_t), dtype=np.uint8)+1,
                ]
                cseq.evidence = strands
            repred_seqs.append(cseq)
        shift += seq.N

    total = total_mis_fwd + total_mis_bwd + total_mis_both
    if len(repred_seqs) > 0:
        log_it(
            "Mismatches: "
            f"{total_mis_fwd} (+) | {total_mis_bwd} (-) | "
            f"{total_mis_both} (+/-). "
            f"Repredicting {total} chunks."
        )

        if repredict_func is not None:
            repred_fwd, repred_bwd = repredict_func(Fasta(repred_seqs))
        else:
            repred_fwd, repred_bwd = predict_func(Fasta(repred_seqs))

        log_it("Merging repredictions.")

    failed_fwd = {}
    failed_bwd = {}
    for k, i in enumerate(repred_index):
        strand = repred_strand[k]
        if strand == 0 or strand == 2:
            labels_fwd[i], labels_fwd[i+1], success = (  # type: ignore
                _merge_replace_center(
                    labels_fwd[i],  # type: ignore
                    labels_fwd[i+1],  # type: ignore
                    repred_fwd[k],  # type: ignore
                )
            )
            if not success:
                if repred_sequence[k] not in failed_fwd:
                    failed_fwd[repred_sequence[k]] = []
                failed_fwd[repred_sequence[k]].append((i+1) * fasta.T)
        if strand == 1 or strand == 2:
            labels_bwd[i], labels_bwd[i+1], success = (  # type: ignore
                _merge_replace_center(
                    labels_bwd[i],  # type: ignore
                    labels_bwd[i+1],  # type: ignore
                    repred_bwd[k],  # type: ignore
                )
            )
            if not success:
                if repred_sequence[k] not in failed_bwd:
                    failed_bwd[repred_sequence[k]] = []
                failed_bwd[repred_sequence[k]].append((i+1) * fasta.T)

    if len(failed_fwd) + len(failed_bwd) > 0:
        log_it(
            "Merging failed for: "
            f"{len(failed_fwd)} (+) | {len(failed_bwd)} (-). "
            f"Setting the areas to intergenic regions."
        )
        shift = 0
        for seq_i, seq in enumerate(fasta):
            if seq_i in failed_fwd:
                lbl_seq = labels_fwd[shift:shift+seq.N, :].flatten()
                for pos in failed_fwd[seq_i]:
                    l = pos - repred_t
                    r = pos + repred_t
                    first_ir_l = _first_non_zero(lbl_seq[:l][::-1] == 0)
                    first_ir_r = _first_non_zero(lbl_seq[r:] == 0)
                    l = 0 if first_ir_l == -1 else l - first_ir_l
                    r = None if first_ir_r == -1 else r + first_ir_r
                    lbl_seq[l:r] = 0
                labels_fwd[shift:shift+seq.N, :] = lbl_seq.reshape(-1, fasta.T)
            if seq_i in failed_bwd:
                lbl_seq = labels_bwd[shift:shift+seq.N, :].flatten()
                for pos in failed_bwd[seq_i]:
                    l = pos - repred_t
                    r = pos + repred_t
                    first_ir_l = _first_non_zero(lbl_seq[:l][::-1] == 0)
                    first_ir_r = _first_non_zero(lbl_seq[r:] == 0)
                    l = 0 if first_ir_l == -1 else l - first_ir_l
                    r = None if first_ir_r == -1 else r + first_ir_r
                    lbl_seq[l:r] = 0
                labels_bwd[shift:shift+seq.N, :] = lbl_seq.reshape(-1, fasta.T)
            shift += seq.N

    log_it("Forming transcripts.")

    entries_fwd: dict[str, list[list[Region]]] = {}
    entries_bwd: dict[str, list[list[Region]]] = {}
    shift = 0
    for seq in fasta:
        if labels_fwd is not None:
            regions = _split_regions(
                labels_fwd[shift:shift+seq.N, :].flatten()[:seq.size],
                strand="+",
            )
            if seq.name not in entries_fwd: entries_fwd[seq.name] = []
            entries_fwd[seq.name] += _transcripts_from_regions(regions)

        if labels_bwd is not None:
            regions = _split_regions(
                labels_bwd[shift:shift+seq.N, :].flatten()[:seq.size],
                strand="-",
            )
            if seq.name not in entries_bwd: entries_bwd[seq.name] = []
            entries_bwd[seq.name] += _transcripts_from_regions(regions)

        shift += seq.N

    log_it("Creating annotation.")
    annotation, last_tx_id = _annotation_from_dict(
        entries_fwd,
        entries_bwd,
        first_tx_id=first_tx_id,
    )
    return annotation, last_tx_id


def _annotate_genome(
    fasta: Path | str,
    predict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ],
    output: Path | str,
    T_max: int = 500_000,
    T_delta: float = 0.1,
    T_factors: list[int] | None = None,
    group_size_limit: int | None = 1_000_000_000,
    model_name: str = "bricks2marble",
    min_sequence_size: int | None = 1_000,
    include_seqs: Container[str] | None = None,
    exclude_seqs: Container[str] | None = None,
    split_seqnames: bool = True,
    repredict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ] | None = None,
    reprediction_factor: float = 0.5,
    repredict_exon_at_boundary: int | None = None,
    concat_strand_to_reprediction: bool = False,
    postprocess: Callable[[Fasta, Annotation], Annotation] | None = None,
) -> None:
    wrapper = textwrap.TextWrapper(
        width=99,
        initial_indent=" " * 13,
        subsequent_indent=" " * 13,
        break_long_words=False,
        break_on_hyphens=False,
    )

    first_tx_id = 0
    for i, group in enumerate(iterate_sequences(
        fasta,
        T_max=T_max,
        delta=T_delta,
        T_factors=T_factors,
        group_size_limit=group_size_limit,
        min_sequence_size=min_sequence_size,
        include=include_seqs,
        exclude=exclude_seqs,
        split_name=split_seqnames,
        log=True,
    )):
        log_it(
            f"\n{f'Group {i+1}':=^99}\n"
                f"> sequences: {len(group)}\n"
                f"{wrapper.fill(', '.join([s.name for s in group]))}\n"
                f"> chunk length: {group.T}\n",
            extra={"timer": False},
        )
        group_annotation, last_tx_id = _annotate(
            group,
            predict_func=predict_func,
            repredict_func=repredict_func,
            reprediction_factor=reprediction_factor,
            concat_strand_to_reprediction=concat_strand_to_reprediction,
            exon_at_boundary=repredict_exon_at_boundary,
            first_tx_id=first_tx_id,
        )
        if postprocess is not None:
            log_it("Calling postprocessing function.")
            group_annotation = postprocess(group, group_annotation)
        log_it("Writing annotation to file.")
        convert(group_annotation, output, append=True)
        log_it("Done.")
        first_tx_id = last_tx_id


def annotate_genome(
    fasta: Path | str,
    predict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ],
    output: Path | str,
    log_file: Path | str | None = None,
    allow_extract_gz: bool = False,
    model_name: str = "bricks2marble",
    T_max: int = 500_000,
    T_delta: float = 0.1,
    T_factors: list[int] | None = None,
    group_size_limit: int | None = 1_000_000_000,
    min_sequence_size: int | None = 1_000,
    split_seqnames: bool = True,
    include_seqs: Container[str] | None = None,
    exclude_seqs: Container[str] | None = None,
    repredict_func: Callable[[Fasta],
        tuple[np.ndarray, np.ndarray | None]
        | tuple[np.ndarray | None, np.ndarray]
        | tuple[np.ndarray, np.ndarray],
    ] | None = None,
    reprediction_factor: float = 0.5,
    repredict_exon_at_boundary: int | None = None,
    concat_strand_to_reprediction: bool = False,
    postprocess: Callable[[Fasta, Annotation], Annotation] | None = None,
    log_config: list[str] | None = None,
) -> None:
    """Generate a genome annotation of a given fasta file.

    The fasta file is read in groups of sequences which total size can
    be specified. The annotation is written on-the-fly.

    Args:
        fasta (Path | str): Path to a fasta file.
        predict_func (Callable): A function that takes a :class:`Fasta`
            object as input and outputs one or two numpy arrays. Each of
            these arrays have to have shape ``(N, T)``, where ``N`` is
            the number of sequences in the given `fasta` and ``T`` is
            its chunk size. The output should be integers of states from
            the :class:`bricks2marble.tf.HMM`. The first numpy array is
            a prediction on the forward strand, the second a prediction
            on the reverse strand. One of them can be missing.
        output (Path | str): Path for the output annotation file. Raises
            an error if the file already exists. Possible file formats
            are ".gtf", ".gff3" and ".gp".
        log_file (Path | str, optional): The file to write a log to.
            Defaults to the same file path as `output`, except with the
            suffix being replaced by `.log`.
        allow_extract_gz (bool, optional): If set to True, allows the
            input path to be a `.gz` file. For the duration of the
            annotation, this file will be extracted to a `.fa` file at
            the same location with a random name. This ensures that the
            same `.gz` file can be annotated multiple times at once, for
            example in a cluster. Defaults to False.
        model_name (str, optional): Name of the model that is used, or
            any other identifier. This will be listed as the 'source' in
            the gtf. Defaults to "bricks2marble".
        T_max (int, optional): Size of chunks in the Fasta given to
            `predict_func`. If all sequences in a group are smaller than
            `T_max`, the chunk size for that group is determined by the
            largest sequence in the group. Defaults to `500_000`.
        T_delta (float, optional): The value `T_max * T_delta` is the
            minimal size of sequences in the first group. Subsequent
            groups have sequences larger than `T * T_delta`, where `T`
            is the chunk length of this group. Defaults to 0.1.
        T_factors (list[int], optional): Imposes extra conditions on
            newly chosen values for the chunk length in case that all
            sequences in a group are smaller than `T_max`. A candidate
            needs to also be divisible by the given numbers. Defaults to
            no such conditions.
        group_size_limit (int, optional): If specified, limits the
            number of nucleotides in each group. Once the limit is
            surpassed, a new group is started with the same chunk size.
            Defaults to 1_000_000_000.
        min_sequence_size (int, optional): If specified, does not return
            sequences with a lower number of nucleotides. Defaults to
            1000.
        split_seqnames (bool, optional): If True, shortens all sequence
            names by cutting at the first whitespace. Defaults to True.
        include_seqs (Container[str], optional): Container of sequence
            names to include in the annotation. The sequence names are
            expected to be also split at the first whitespace, if
            `split_seqnames` is True. Defaults to all sequences.
        exclude_seqs (Container[str], optional): Container of sequence
            names to exclude in the annotation. The sequence names are
            expected to be also split at the first whitespace, if
            `split_seqnames` is True. If a name is in `include_seqs` and
            `exclude_seqs`, it will be excluded. Defaults to no excluded
            sequences.
        repredict_func (Callable, optional): A function of the same
            nature as `predict_func`, but used for the second stage of
            predictions that solves mismatches of the first. The default
            is that `predict_func` is used for repredictions, too.
        reprediction_factor (float, optional): A float between 0 and 1
            that determines the length of chunks used in the
            reprediction. A value of 1 means that the size is twice the
            size of the first predictions.
        repredict_exon_at_boundary (int, optional): If the model
            predicts an exon within ``k`` positions left and right of a
            break point, a reprediction will be made. This works only if
            ``liberal=True``. Here, ``k`` is the given number for this
            argument. Defaults to no additional checks for exons.
        concat_strand_to_reprediction (bool, optional): If True, appends
            a numpy array of integers `{0, 1, 2}` as a `Fasta.evidence`
            entry to the sequences that are given to the reprediction
            function. A `0` means that this sequence is for the forward
            strand, `1` for the backward strand and `2` for both.
            Defaults to False.
        postprocess (callable, optional): An optional function that
            changes each annotation before writing it to the gtf file.
            Useful for cleaning up errors from the prediction. The
            function takes a :class:`Fasta` and an :class:`Annotation`
            object as arguments and returns an :class:`Annotation`
            object.
        log_config (list[str], optional): A list of strings to
            write to the `log_file` as additional information at the
            start of the annotation. Defaults to None.
    """
    fasta = Path(fasta)
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"The target output path {output} points to an existing file."
        )
    allowed(Annotation(), output)
    if log_file is None: log_file = output.parent / f"{output.stem}.log"

    setup_logging(log_file)

    log_it(
        f"{'':-^99}\n|{f'{model_name} genome annotation': ^97}|\n{'':-^99}\n",
        extra={"timer": False},
    )
    config_log = [f"target genome file: {fasta}"]
    if include_seqs is not None:
        config_log += [f"include sequences: {include_seqs}"]
    if exclude_seqs is not None:
        config_log += [f"exclude sequences: {exclude_seqs}"]
    log_reab = (
        'None' if repredict_exon_at_boundary is None
        else repredict_exon_at_boundary
    )
    config_log += [
        f"output annotation file: {output}",
        f"maximal chunk length: {T_max}",
        f"forced divisors of the chunk length: {T_factors}",
        f"minimal sequence size: {min_sequence_size}",
        f"group size limit: {group_size_limit}",
        f"reprediction factor: {reprediction_factor}",
        f"repredict exons near boundaries: {log_reab}",
        f"postprocessing: {postprocess is not None}",
    ]
    if log_config is not None:
        config_log += log_config
    log_it("> " + "\n> ".join(config_log) + "\n", extra={"timer": False})

    def _call_annotate(fasta_file: Path | str):
        _annotate_genome(
            fasta=fasta_file,
            predict_func=predict_func,
            output=output,
            T_max=T_max,
            T_delta=T_delta,
            T_factors=T_factors,
            group_size_limit=group_size_limit,
            model_name=model_name,
            min_sequence_size=min_sequence_size,
            include_seqs=include_seqs,
            exclude_seqs=exclude_seqs,
            split_seqnames=split_seqnames,
            repredict_func=repredict_func,
            reprediction_factor=reprediction_factor,
            repredict_exon_at_boundary=repredict_exon_at_boundary,
            concat_strand_to_reprediction=concat_strand_to_reprediction,
            postprocess=postprocess,
        )

    if fasta.suffix == ".gz":
        if not allow_extract_gz: raise RuntimeError(
            "Given fasta file is a .gz archive but extracting is not allowed."
        )
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            log_it(
                "Given file is an archive and has to be extracted.\n"
                f"> using temporary file location:\n\t{tmp.name}",
                extra={"timer": False},
            )
            with gzip.open(fasta, "rb") as f_fasta_gz:
                shutil.copyfileobj(f_fasta_gz, tmp)

            fasta = Path(tmp.name)
            _call_annotate(fasta)
        return
    _call_annotate(fasta)

    log_it(
        f"\n{'':-^99}\n|{'Finished': ^97}|\n{'':-^99}",
        extra={"timer": False},
    )
