from pathlib import Path

import numpy as np

from .fasta import Sequence


class IndexedFasta:
    """Class of an indexed Fasta object that only loads required
    sequence parts into memory.
    Only works on unzipped fasta objects.
    """

    def __init__(self, path: Path | str) -> None:
        from ..io import index

        self._path = Path(path)
        if self._path.suffix == ".gz":
            raise RuntimeError(
                "IndexedFasta on zipped files is not supported."
            )
        self.index = {s[0]: s[1:] for s in index(self._path)}
        self._L = None

        table = bytearray([4]*256)
        mappings = {a: b for a, b in zip(
            b"ACGTNnacgt",
            [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
        )}
        for k, v in mappings.items():
            table[k] = v
        self.translation_table = bytes(table)

    @property
    def L(self) -> int:
        """Length of one line of nucleotides in the fasta file. This is
        assumed to be constant throughout the file.
        """
        if self._L is None: self._L = self._detect_line_length()
        return self._L

    def _detect_line_length(self) -> int:
        with open(self._path, "rb") as f:
            f.seek(next(iter(self.index.values()))[0])
            line = f.readline()
        return len(line.rstrip(b"\n\r"))

    def _nuc_to_byte(self, seq_byte_start: int, i: int) -> int:
        """Convert nucleotide position i to file byte offset."""
        return seq_byte_start + i + (i // self.L)

    def fetch(self, seq_name: str, *range_: tuple[int, int]) -> Sequence:
        """Return nucleotides from one or more ranges `[a, b]` from
        sequence `seq_name`. Indexing starts at 0 and is end-exclusive.
        """
        if seq_name not in self.index:
            raise KeyError(f"Sequence '{seq_name}' not in index")

        idx = self.index[seq_name]
        seq_byte_start, _, seq_len = idx

        raw_all = b""
        for a, b in range_:
            if a < 0 or b > seq_len or a > b:
                raise ValueError(
                    f"Range [{a}, {b}] out of bounds for sequence "
                    f"'{seq_name}' of length {seq_len}."
                )

            byte_a = self._nuc_to_byte(seq_byte_start, a)
            byte_b = self._nuc_to_byte(seq_byte_start, b)

            with open(self._path, "rb") as f:
                f.seek(byte_a)
                raw = f.read(byte_b - byte_a)
            raw_all += raw.replace(b"\n", b"").replace(b"\r", b"")

        name = seq_name+":"+",".join(map(lambda x: f"{x[0]+1}-{x[1]}", range_))
        return Sequence(
            np.frombuffer(
                raw_all.translate(self.translation_table),
                dtype=np.int8,
            )[np.newaxis, :],
            name=name,
        )
