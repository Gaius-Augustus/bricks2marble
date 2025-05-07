from pathlib import Path

import numpy as np

from ..struct.fasta import FASTA, MAP, Segment


def load_fasta(
    path: Path | str,
    T: int,
    restrict: int = -1,
    repeat_masking: bool = True,
    overlap: int = 0,
    use_map: dict[str, int] | None = None,
) -> FASTA:
    """Loads a :class:`FASTA` object that makes handling a nucleotide
    sequence easier.

    Args:
        path (Path | str): Path to the fasta file.
        T (int): The whole genome is split into smaller sequence chunks
            of this size. Sequences in the file that have a length not
            divisible by ``T`` are padded with ``-1``.
        restrict (int, optional): Restrict the reading window. Only
            reads the given number of nucleotides from the file.
            Defaults to -1, which means everything is read.
        repeat_masking (bool, optional): Whether to differentiate
            repeat-masked nucleotides and normal ones in the file.
            Defaults to True.
        overlap (int, optional): If greater than zero, two consecutive
            sequences overlap by the given integer. Defaults to 0.
        use_map (dict[str, int], optional): The encoding to use for the
            nucleotides. The default order of enumeration is ``A C G T N
            a c g t``.
    """
    with open(path, "r") as f:
        lines = f.readlines(restrict)

    if use_map is None:
        use_map = MAP
    else:
        use_map = use_map.copy()
        if "n" not in use_map:
            use_map["n"] = use_map["N"]

    sequences: list[str] = []
    name_sequences: list[str] = []
    for line in lines:
        if line.startswith(">"):
            name_sequences.append(line[1:].strip())
            sequences.append("")
        else:
            if repeat_masking:
                sequences[-1] += line.strip()
            else:
                sequences[-1] += line.strip().upper()

    sequences_ = []
    for seq in sequences:
        sequences_.append(np.array(list(map(use_map.get, seq))))
    del sequences

    all_sequences = []
    coords: list[Segment] = []
    for k, seq in enumerate(sequences_):
        N, left = divmod(len(seq) - overlap, T - overlap)
        enc = np.zeros((N + int(left>0), T), dtype=np.int8)
        T_sample = T - overlap
        for i in range(N):
            enc[i, :] = seq[i * T_sample : i * T_sample + T]
            coords.append(Segment(
                name=name_sequences[k],
                start=i * T_sample + 1,
                end=i * T_sample + T,
            ))
        if left > 0:
            surplus = left + overlap
            if surplus > T:
                raise RuntimeError("Dataset creation failed")
            enc[-1, :surplus] = seq[-surplus:]
            enc[-1, surplus:] = -1
            coords.append(Segment(
                name=name_sequences[k],
                start=N * T_sample + 1,
                end=N * T_sample + surplus,
            ))
        all_sequences.append(enc)

    return FASTA(np.concatenate(all_sequences, 0), coords)
