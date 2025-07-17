from pathlib import Path

import numpy as np

from ..struct.fasta import FASTA, Sequence


def load_fasta(
    path: Path | str,
    T: int | None = None,
    drop_remainder: bool = False,
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
        drop_remainder (bool, optional): If set to True, deletes the
            last chunk in each sequence, if the sequence has a length
            not divisable by ``T``. Defaults to False.
        restrict (int, optional): Restrict the reading window. Only
            reads the given number of nucleotides from the file.
            Defaults to -1, which means everything is read.
    """
    with open(path, "r", encoding="utf-8") as f:
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
        b"ACGTNnacgt",
        bytes([0, 1, 2, 3, 4, 4, 5, 6, 7, 8]),
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
        fasta.resample(T, drop_remainder=drop_remainder)
    return fasta


def write_fasta(
    fasta: FASTA,
    path: Path| str,
    line_length: int = 80,
) -> None:
    path = Path(path)
    if path.is_file() and path.exists():
        raise FileExistsError(f"The file {path} already exists.")
    if path.is_dir():
        raise IsADirectoryError(
            f"The given path {path} points to a directory but should be a "
            "file."
        )

    translation_table = bytes.maketrans(
        bytes([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        b"ACGTNacgt",
    )
    with open(path, "w") as f:
        for sequence in fasta:
            f.write(f">{sequence.name}\n")
            translated = sequence.flat.tobytes().translate(translation_table)
            translated = translated.decode("utf-8")
            for i in range(0, len(translated), line_length):
                f.write(translated[i:i+line_length]+"\n")
