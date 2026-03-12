import gzip
from collections.abc import Container, Generator
from pathlib import Path

import numpy as np

from ..log import log_it
from ..struct.fasta import Fasta, Sequence
from .util import index, largest_close_to_divisible_by


def load_fasta(
    path: Path | str,
    T: int | None = None,
    include: Container[str] | None = None,
    exclude: Container[str] | None = None,
    split_name: bool = True,
) -> Fasta:
    """Loads a :class:`Fasta` object that makes handling a nucleotide
    sequence easier.

    Can be either a fasta or a gzipped fasta file (suffix `.gz`). A
    `.gz` file loads much slower than its unzipped counterpart.

    Args:
        path (Path | str): Path to the fasta file.
        T (int, optional): The whole genome is split into smaller
            sequence chunks of this size. Sequences in the file that
            have a length not divisible by ``T`` are padded with ``-1``.
            Defaults to the total length of the genome, meaning that the
            Fasta file only contains one long sequence.
        include (Container[str], optional): Container of sequence names
            to include. The sequence names are expected to be also split
            at the first whitespace, if `split_name` is True. Defaults
            to all sequences.
        exclude (Container[str], optional): Container of sequence names
            to exclude. The sequence names are expected to be also split
            at the first whitespace, if `split_name` is True. If a name
            is in `include` and `exclude`, it will be excluded. Defaults
            to no excluded sequences.
        split_name (bool, optional): If True, cuts the name of all
            sequences at the first whitespace. Defaults to True.
    """
    if Path(path).suffix == ".gz":
        fasta = _load_gz_fasta(
            path,
            include=include,
            exclude=exclude,
            split_name=split_name,
        )
    else:
        fasta = Fasta([s[0] for s in iterate_sequences(
            path,
            include=include,
            exclude=exclude,
            split_name=split_name,
        )])

    if T is not None: fasta.resample(T)
    return fasta


def write_fasta(
    fasta: Fasta,
    path: Path| str,
    line_length: int = 80,
    as_gz: bool = False,
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
    open_func = gzip.open if as_gz else open
    with open_func(path, "wt") as f:
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
    fasta: Path | str,
    T_max: int | None = None,
    delta: float = 0.1,
    T_factors: list[int] | None = None,
    group_size_limit: int | None = None,
    min_sequence_size: int | None = None,
    sort_reverse: bool | None = None,
    include: Container[str] | None = None,
    exclude: Container[str] | None = None,
    split_name: bool = True,
    log: bool = False,
) -> Generator[Fasta, None, None]:
    """Yields Fasta objects from the given fasta file. This can be
    helpful when processing sequences with a limited ammount of RAM.
    Each returned Fasta is a group of whole sequences from the given
    file.

    If `T_max` is specified, the sequences are sorted by length before
    returned. Smaller sequences will be put into their own groups.

    Args:
        fasta (Path): The path to a fasta to iterate over.
        T_max (int, optional): If given, orders sequences by length and
            resamples all sequences to the given chunksize. If all
            sequences in a group are smaller than `T_max`, the chunk
            size for that group is determined by the largest sequence in
            the group. Defaults to no action.
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
            Only supported for `T_max != None`. Defaults to no limit.
        min_sequence_size (int, optional): If specified, does not return
            sequences with a lower number of nucleotides. Defaults to
            returning all sequence lengths.
        sort_reverse (bool, optional): Whether to sort sequences in
            descending order (True) or ascending order (False). Defaults
            to not sorting sequences if `T_max` is not specified, else
            `True`.
        include (Container[str], optional): Container of sequence names
            to include. The sequence names are expected to be also split
            at the first whitespace, if `split_name` is True. Defaults
            to all sequences.
        exclude (Container[str], optional): Container of sequence names
            to exclude. The sequence names are expected to be also split
            at the first whitespace, if `split_name` is True. If a name
            is in `include` and `exclude`, it will be excluded. Defaults
            to no excluded sequences.
        split_name (bool, optional): If True, cuts the name of all
            sequences at the first whitespace. Defaults to True.
        log (bool, optional): Wether to log process. Only applicable if
            wrapped in another function that starts a logging process.
            Defaults to False.
    """
    fasta = Path(fasta)
    idx = index(
        fasta,
        sort_reverse=True if T_max is not None else sort_reverse,
        min_sequence_size=min_sequence_size,
        include=include,
        exclude=exclude,
        split_name=split_name,
    )

    table = bytearray([4]*256)
    mappings = {a: b for a, b in zip(
        b"ACGTNnacgt",
        [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
    )}
    for k, v in mappings.items():
        table[k] = v
    translation_table = bytes(table)

    if T_max is None:
        groups = list(range(len(idx)+1))
    else:
        groups = [0]
        group_T = [T_max]
        i = 0
        T = T_max
        while i < len(idx):

            gs = 0
            group_overflow = False
            while i < len(idx) and (
                idx[i][3] >= (delta * T)
                or (T_factors is not None and idx[i][3] < min(T_factors))
            ):
                if T_factors is None or idx[i][3] >= min(T_factors):
                    last_len = idx[i][3]
                gs += idx[i][3]
                i += 1
                if group_size_limit is not None and gs > group_size_limit:
                    group_overflow = True
                    break

            if group_overflow:
                T_new = T
            else:
                T_new = last_len if T_factors is None else (
                    largest_close_to_divisible_by(last_len, T_factors)
                )
            if i > groups[-1]:
                groups.append(i)
                group_T.append(T)
            T = T_new

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
        with open(fasta, "rb") as f:
            for k in range(groups[g-1], groups[g]):
                f.seek(idx[k][1])
                seq = f.read(idx[k][2] - idx[k][1])
                raw_sequences.append(seq.replace(b"\n", b""))
                name_sequences.append(idx[k][0])

        group = Fasta([Sequence(
            np.frombuffer(
                seq.translate(translation_table),
                dtype=np.int8,
            )[np.newaxis, :],
            name=name,
        ) for seq, name in zip(raw_sequences, name_sequences)])

        if T_max is not None: group.resample(group_T[g])

        yield group


def _load_gz_fasta(
    path: Path | str,
    include: Container[str] | None = None,
    exclude: Container[str] | None = None,
    split_name: bool = True,
) -> Fasta:
    raw_sequences: list[bytes] = []
    name_sequences: list[str] = []
    current_name = None
    current_buffer = None
    collecting = include is None and exclude is None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if current_name is not None and collecting:
                    raw_sequences.append(bytes(current_buffer))
                    name_sequences.append(current_name)

                current_name = line[1:].strip()
                if split_name: current_name = current_name.split(" ")[0]
                collecting = (
                    (include is None or current_name in include)
                    and
                    (exclude is None or current_name not in exclude)
                )
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
    return fasta
