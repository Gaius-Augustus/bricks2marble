import gzip
from collections.abc import Generator
from pathlib import Path

import numpy as np

from ..log import log_it
from ..struct.fasta import Fasta, Sequence
from .util import index, largest_close_to_divisible_by


def load_fasta(
    path: Path | str,
    T: int | None = None,
    drop_remainder: bool = False,
    n_seqs: int | None = None,
    target_seq: str | None = None,
) -> Fasta:
    """Loads a :class:`Fasta` object that makes handling a nucleotide
    sequence easier. Can be either a fasta file or a gzipped fasta file.

    Args:
        path (Path | str): Path to the fasta file.
        T (int, optional): The whole genome is split into smaller
            sequence chunks of this size. Sequences in the file that
            have a length not divisible by ``T`` are padded with ``-1``.
            Defaults to the total length of the genome, meaning that the
            Fasta file only contains one long sequence.
        drop_remainder (bool, optional): If set to True, deletes the
            last chunk in each sequence, if the sequence has a length
            not divisable by ``T``. Defaults to False.
        n_seqs (int, optional): Stops reading the fasta file after this
            many sequences. Defaults to reading all sequences.
        target_seq (str, optional): Scans the Fasta file from the start
            up to the given sequence name and then only returns this
            sequence. Defaults to all sequences.
    """
    if target_seq is not None and n_seqs is not None:
        raise ValueError(
            "Specifying both 'n_seqs' and 'target_seq' is not allowed."
        )

    raw_sequences: list[bytes] = []
    name_sequences: list[str] = []
    current_name = None
    current_buffer = None
    collecting = target_seq is None
    open_func = gzip.open if Path(path).suffix == ".gz" else open
    with open_func(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if current_name is not None and collecting:
                    raw_sequences.append(bytes(current_buffer))
                    name_sequences.append(current_name)

                    if target_seq is not None:
                        break
                    if n_seqs is not None and len(name_sequences) >= n_seqs:
                        break

                current_name = line[1:].strip()
                collecting = target_seq is None or current_name == target_seq
                current_buffer = bytearray() if collecting else None
            else:
                if collecting: current_buffer.extend(line.strip().encode())
        else:
            if current_name is not None and collecting:
                raw_sequences.append(bytes(current_buffer))
                name_sequences.append(current_name)

    table = bytearray([4]*256)
    mappings = {a: b for a, b in zip(
        b"ACGTNnacgt",
        [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
    )}
    for k, v in mappings.items():
        table[k] = v
    translation_table = bytes(table)

    sequences = [
        Sequence(
            np.frombuffer(
                seq.translate(translation_table),
                dtype=np.int8,
            )[np.newaxis, :],
            name=name,
        ) for seq, name in zip(raw_sequences, name_sequences)
    ]
    fasta = Fasta(sequences)
    if T is not None:
        fasta.resample(T, drop_remainder=drop_remainder)
    return fasta


def write_fasta(
    fasta: Fasta,
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


def fasta_from_string(string: str) -> Fasta:
    """Creates a :class:`Fasta` object with a single sequence from a
    continuous string of nucleotides.
    """
    table = bytearray([4]*256)
    mappings = {a: b for a, b in zip(
        b"ACGTNnacgt",
        [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
    )}
    for k, v in mappings.items():
        table[k] = v
    translation_table = bytes(table)

    sequences = [Sequence(
        np.frombuffer(
            string.encode().translate(translation_table),
            dtype=np.int8,
        )[np.newaxis, :],
        name="seq1",
    )]
    return Fasta(sequences)


def iterate_sequences(
    fasta: Path,
    min_group_size: int = 50_000_000,
    T_max: int | None = None,
    T_factors: list[int] | None = None,
    sort_reverse: bool | None = False,
    log: bool = False,
) -> Generator[Fasta, None, None]:
    """Yields Fasta objects from the given fasta file in a sorted
    manner. This can be helpful when processing sequences with a
    limited ammount of RAM.

    Each returned Fasta is a group of whole sequences from the given
    file. This leads to large sequences being returned on their own and
    small sequences being grouped together.

    Args:
        min_group_size (int, optional): Minimal number of nucleotides in
            each group except for the last, which can be smaller.
            Defaults to `50_000_000`.
        T_max (int, optional): If given, also resamples all sequences to
            the given chunksize. If all sequences in a group are smaller
            than `T_max`, the chunk size for that group is determined by
            the largest sequence in the group. Defaults to no
            resampling.
        T_factors (list[int], optional): Imposes extra conditions on
            newly chosen values for the chunk length in case that all
            sequences in a group are smaller than `T_max`. A candidate
            needs to also be divisible by the given numbers. Defaults to
            no such conditions.
        sort_reverse (bool, optional): Whether to sort the sequences by
            size before grouping them. If true, sorts in descending
            order and if false, sorts in ascending order. Set to None
            for no sorting. Defaults to False.
    """
    idx = index(fasta, sort_reverse=sort_reverse)

    table = bytearray([4]*256)
    mappings = {a: b for a, b in zip(
        b"ACGTNnacgt",
        [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
    )}
    for k, v in mappings.items():
        table[k] = v
    translation_table = bytes(table)

    groups = [0]
    i = 0
    while i < len(idx):
        gs = 0
        while gs < min_group_size and i < len(idx):
            gs += idx[i][3] if T_max is None or T_max < idx[i][3] else T_max
            i += 1
        groups.append(i)

    if log:
        seqs = "" if len(idx) == 1 else "s"
        gros = "" if len(groups) == 2 else "s"
        log_it(
            f"Split {len(idx)} sequence{seqs} into "
                f"{len(groups)-1} group{gros}.",
            extra={"timer": False},
        )

    for g in range(1, len(groups)):
        raw_sequences = []
        name_sequences = []
        max_len = 0
        with open(fasta, "rb") as f:
            for k in range(groups[g-1], groups[g]):
                f.seek(idx[k][1])
                seq = f.read(idx[k][2] - idx[k][1])
                raw_sequences.append(seq.replace(b"\n", b""))
                name_sequences.append(idx[k][0])
                if idx[k][3] > max_len: max_len = idx[k][3]

        group = Fasta([Sequence(
            np.frombuffer(
                seq.translate(translation_table),
                dtype=np.int8,
            )[np.newaxis, :],
            name=name,
        ) for seq, name in zip(raw_sequences, name_sequences)])

        if T_max is not None:
            T = T_max
            if max_len < T_max:
                T = max_len if T_factors is None else (
                    largest_close_to_divisible_by(max_len, T_factors)
                )
            group.resample(T)

        yield group
