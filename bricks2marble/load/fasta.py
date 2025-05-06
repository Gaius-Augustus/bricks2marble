from pathlib import Path

import numpy as np

from ..struct.fasta import ENCODING, FASTA, MAP, Region


def load_fasta(
    path: Path | str,
    T: int,
    restrict: int = -1,
    repeat_masking: bool = True,
    overlap: int = 0,
    pad: int = -1,
    encoding: np.ndarray | None = None,
    use_map: dict[str, int] | None = None,
) -> FASTA:
    """Loads a :class:`FASTA` object that includes the one-hot encoded
    sequences of nucleotides and
    """
    with open(path, "r") as f:
        lines = f.readlines(restrict)

    if encoding is None:
        encoding = ENCODING[:5] if not repeat_masking else ENCODING
    if use_map is None:
        use_map = MAP

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

    encoding = encoding
    sequences_ = []
    for seq in sequences:
        sequences_.append(np.array(list(map(use_map.get, seq))))
    del sequences

    all_sequences = []
    coords: list[Region] = []
    for k, seq in enumerate(sequences_):
        N, left = divmod(len(seq) - overlap, T - overlap)
        enc = np.zeros((N + int(left>0), T), dtype=np.int8)
        T_sample = T - overlap
        for i in range(N):
            enc[i, :] = seq[i * T_sample : i * T_sample + T]
            coords.append(Region(
                name=name_sequences[k],
                start=i * T_sample + 1,
                end=i * T_sample + T,
            ))
        if left > 0:
            surplus = left + overlap
            if surplus > T:
                raise RuntimeError("Dataset creation failed")
            enc[-1, :surplus] = seq[-surplus:]
            enc[-1, surplus:] = pad
            coords.append(Region(
                name=name_sequences[k],
                start=N * T_sample + 1,
                end=N * T_sample + surplus,
            ))
        all_sequences.append(enc)

    return FASTA(np.concatenate(all_sequences, 0), coords)
