from pathlib import Path

import numpy as np

from ..struct.fasta import FASTA, Sequence


def load_fasta(
    path: Path | str,
    T: int | None = None,
    restrict: int = -1,
) -> FASTA:
    """Loads a :class:`FASTA` object that makes handling a nucleotide
    sequence easier.

    Args:
        path (Path | str): Path to the fasta file.
        T (int, optional): The whole genome is split into smaller
            sequence chunks of this size. Sequences in the file that
            have a length not divisible by ``T`` are padded with ``-1``.
            Defaults to the total length of the genome, meaning that the
            FASTA file only contains one long sequence.
        restrict (int, optional): Restrict the reading window. Only
            reads the given number of nucleotides from the file.
            Defaults to -1, which means everything is read.
    """
    with open(path, "r") as f:
        lines = f.readlines(restrict)

    raw_sequences: list[bytes] = []
    name_sequences: list[str] = []
    current_sequence = ""
    for line in lines:
        if line.startswith(">"):
            if current_sequence:
                raw_sequences.append(current_sequence.encode())
                current_sequence = ""
            name_sequences.append(line[1:].strip())
        else:
            current_sequence += line.strip()
    raw_sequences.append(current_sequence.encode())

    translation_table = bytes.maketrans(
        b"ACGTNacgt",
        bytes([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    )
    sequences = []
    for seq, name in zip(raw_sequences, name_sequences):
        translated = seq.translate(translation_table)
        sequences.append(Sequence(
            np.frombuffer(translated, dtype=np.int8)[np.newaxis, :],
            name=name,
        ))
    fasta = FASTA(sequences)
    if T is not None:
        fasta = fasta.resample(T)
    return fasta
