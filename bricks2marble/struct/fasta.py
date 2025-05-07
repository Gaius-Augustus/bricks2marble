from typing import Literal

import numpy as np
from pydantic import BaseModel

MAP = {
    "A": 0, "C": 1, "G": 2, "T": 3, "N": 4,
    "a": 5, "c": 6, "g": 7, "t": 8, "n": 4,
}
ENCODING = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 1],
])


class Segment(BaseModel):
    """A segment has a name and a defined range. It is typically used to
    determine the origin of a sequence of nucleotides within a greater
    context, e.g. a genome.
    """

    name: str
    start: int
    end: int

    def __eq__(self, other: "Segment | tuple[str, int, int]") -> bool:
        if isinstance(other, tuple):
            return (
                self.name == other[0]
                and self.start == other[1]
                and self.end == other[2]
            )
        return super().__eq__(other)


class Region(Segment):
    """A region is a special type of segment that differs between
    forward and backward strand in a genome. It is typically used to
    mark a feature by type in a genome (exon, intron, ...), without
    specifying further information.
    """

    strand: Literal["+", "-"] = "+"

    def __eq__(self, other) -> bool:
        if isinstance(other, Region):
            return super().__eq__(other)
        raise NotImplementedError("Can only compare Region to Region type")


class FASTA:
    """This class manages a sequence of nucleotides that is grouped into
    chunks of a given length with additional information about their
    origin (sequence name and range).

    It is likely that you never initialize this class by hand but
    instead use the corresponding loading method
    :meth:`bricks2marble.load_fasta`.

    Args:
        sequences (np.ndarray): An array of shape ``(N, T)`` which holds
            encoded nucleotides, where ``N`` is the number of sequences
            of length ``T``. This array is later accessed by
            ``FASTA.nuc``.
        segments (list[Segment]): A list of length ``N``, matching the
            first dimension of ``sequences``. Each element is a
            :class:`Segment` object specifying where each sequence is
            coming from.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        segments: list[Segment],
    ) -> None:
        self._sequences = sequences
        self._segments = segments
        self.repeat_masking = np.any(sequences > 4)

    @property
    def nuc(self) -> np.ndarray:
        """Sequences of encoded nucleotides of shape ``(N, T)``."""
        return self._sequences

    @property
    def segments(self) -> list[Segment]:
        """Sequences of segments of length ``N``."""
        return self._segments

    @property
    def N(self) -> int:
        return self.nuc.shape[0]

    @property
    def T(self) -> int:
        return self.nuc.shape[1]

    def comprises(self) -> dict[str, tuple[int, int]]:
        """Returns a dictionary mapping sequence names to the ranges
        of nucleotides that are in this :class:`FASTA` object.
        """
        names = {c.name for c in self.segments}
        content = {}
        for name in names:
            start = min(c.start for c in self.segments if c.name == name)
            end = max(c.end for c in self.segments if c.name == name)
            content[name] = (start, end)
        return content

    def resample(self, T: int) -> "FASTA":
        """Returns a :class:`FASTA` object that has the same sequences
        grouped into chunks of the given length. This can lead to
        differently padded sequences.
        """
        sequences = []
        new_coords = []

        file_data = {}
        for row, c in zip(self.nuc, self.segments):
            valid_len = c.end - c.start + 1
            data = row[:valid_len]
            if c.name not in file_data:
                file_data[c.name] = []
            file_data[c.name].extend(data)

        for seq, values in file_data.items():
            total = len(values)
            num_chunks = (total + T - 1) // T

            for i in range(num_chunks):
                chunk_start = i * T
                chunk_end = min((i + 1) * T, total)
                chunk = values[chunk_start:chunk_end]

                if len(chunk) < T:
                    chunk += [-1] * (T - len(chunk))

                sequences.append(chunk)
                new_coords.append(Segment(
                    name=seq,
                    start=chunk_start+1,
                    end=chunk_end,
                ))

        return FASTA(np.array(sequences), new_coords)

    def one_hot(self, sequences: np.ndarray | None = None) -> np.ndarray:
        """Returns a one-hot encoded version of :meth:`FASTA.nuc` of
        shape ``(N, T, 5)``. If the sequences are repeat-masked, these
        positions are two-hot vectors and the last dimension is expanded
        by 1, leading to sequences of shape ``(N, T, 6)``.
        """
        seq = self.nuc if sequences is None else sequences
        seq[seq == -1] = 4
        if not self.repeat_masking:
            return ENCODING[:5, :5][seq]
        return ENCODING[seq]

    def occurences(
        self,
        separate_repeat_masked: bool = False,
    ) -> dict[str, float]:
        """Counts the number of occurences of each nucleotide over all
        sequences.

        Args:
            separate_repeat_masked (bool, optional): Whether to separate
                repeat-masked nucleotides from non-masked. Defaults to
                False.
        """
        if self.repeat_masking and separate_repeat_masked:
            probs = {token: 0. for token in MAP.keys() if token != "n"}
        else:
            probs = {token: 0. for token in MAP.keys()
                     if token.upper() == token}
        size = (self.nuc[self.nuc != -1]).size
        if self.repeat_masking:
            for token, index in MAP.items():
                if token == "n": continue
                if not separate_repeat_masked:
                    token = token.upper()
                probs[token] += (self.nuc == index).sum() / size
        else:
            for token, index in MAP.items():
                if token.upper() == token:
                    probs[token] += (self.nuc == index).sum() / size
        return probs

    def positions(
        self,
        start: int | None = None,
        end: int | None = None,
        /,
    ) -> "FASTA":
        """Returns a :class:`FASTA` object that only includes
        nucleotides from the specified range with matching segments. The
        range is specified for ``FASTA.nuc`` without the padding, i.e.
        it does not look up the actual positions of the sequences but
        instead uses its current ordering in the memory of this object.

        Args:
            start (int, optional): First index of the specified range.
            end (int, optional): Last index of the specified range.
        """
        if end is None:
            if start is None:
                start, end = 0, -1
            else:
                end = start
                start = 0
        start = int(start)  # type: ignore

        flat_nuc = []
        flat_seq = []
        flat_pos = []
        for seq, c in zip(self.nuc, self.segments):
            valid_len = c.end - c.start + 1
            flat_nuc.append(seq[:valid_len])
            flat_seq.append(np.full(valid_len, c.name, dtype=object))
            flat_pos.append(np.arange(c.start-1, c.end))
        sliced_nuc = np.concatenate(flat_nuc)[start:end]
        sliced_seq = np.concatenate(flat_seq)[start:end]
        sliced_pos = np.concatenate(flat_pos)[start:end]

        new_seqs = []
        new_coords = []
        ordered_seq = sliced_seq[
            np.sort(np.unique(sliced_seq, return_index=True)[1])
        ]
        for seq in np.unique(ordered_seq):
            mask = (sliced_seq == seq)
            file_vals = sliced_nuc[mask]
            file_pos = sliced_pos[mask]

            num_chunks = -(-len(file_vals) // self.T)
            pad_len = num_chunks * self.T - len(file_vals)

            padded_vals = np.pad(file_vals, (0, pad_len), constant_values=-1)
            padded_pos = np.pad(file_pos, (0, pad_len), constant_values=-1)

            reshaped_vals = padded_vals.reshape(-1, self.T)
            reshaped_pos = padded_pos.reshape(-1, self.T)

            new_seqs.extend(reshaped_vals)
            for row_pos in reshaped_pos:
                real_pos = row_pos[row_pos != -1]
                if len(real_pos) > 0:
                    new_coords.append(Segment(
                        name=seq,
                        start=real_pos[0] + 1,
                        end=real_pos[-1] + 1,
                    ))
                else:
                    new_coords.append(Segment(
                        name=seq,
                        start=-1,
                        end=-1
                    ))

        return FASTA(np.array(new_seqs), new_coords)
