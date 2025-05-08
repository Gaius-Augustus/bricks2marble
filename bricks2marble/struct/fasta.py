from typing import Literal, overload

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


class Sequence:
    """Class representing a single nucleotide sequence or a continuous
    subsequence of it. Each :class:`Sequence` has the name of its origin
    sequence and a start and end index of where the (sub-)sequence is
    found within the context sequence.

    The sequences are encoded in chunks of a particular length ``T`` and
    consist of integers 0 to 8 with the order ``ACGTNacgt``. Optional
    padding is done with a token ``-1`` at the end of the last sequence,
    if necessary.

    Args:
        sequences (list[Sequence]): An array of shape ``(N, T)`` which
            holds encoded nucleotides, where ``N`` is the number of
            sequences of length ``T``. This array is later accessed by
            ``FASTA.nuc``.
    """

    def __init__(
        self,
        sequence: np.ndarray,
        name: str,
        start: int = 0,
        end: int = -1,
    ) -> None:
        self.name = name
        self._sequence = sequence
        self._start = start
        self._end = end

    @property
    def nuc(self) -> np.ndarray:
        return self._sequence

    @property
    def flat(self) -> np.ndarray:
        return self._sequence.flatten()[:self.size]

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def N(self) -> int:
        return self._sequence.shape[0]

    @property
    def T(self) -> int:
        return self._sequence.shape[1]

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        if self._end >= 0:
            return self._end
        else:
            non_padded = np.nonzero(self._sequence[-1] == -1)
            if non_padded[0].size > 0:
                self._end = (self.N-1)*self.T + non_padded[0][0] + 1
            else:
                self._end = self._sequence.size
            return self._end

    def segments(self) -> list[Segment]:
        return [
            Segment(
                name=self.name,
                start=self.start+(i-1)*self.T+1,
                end=min(self.start+i*self.T, self.start+self.size),
            ) for i in range(1, self.N+1)
        ]

    def one_hot(
        self,
        sequences: np.ndarray | None = None,
        pad_index: int = 4,
    ) -> np.ndarray:
        """Returns a one-hot encoded version of :meth:`Sequence.nuc` of
        shape ``(N, T, 6)``.

        Args:
            sequences (np.ndarray, optional): Which sequences to encode.
                Defaults to ``self.nuc``.
            pad_index (int, optional): What to replace the padding
                character (-1) by before encoding. Default to 4, which
                is an `N`.
        """
        nuc = self.nuc if sequences is None else sequences
        nuc[nuc == -1] = pad_index
        return ENCODING[nuc]

    def resample(self, T: int) -> "Sequence":
        """Returns a :class:`Sequence` that is grouped into chunks of
        the given length. This can lead to differently padded sequences.
        """
        if T <= 0:
            raise ValueError(f"Unallowed chunk size ({T})")
        missing = (-self.size) % T
        N = (self.size + missing) // T
        flattened = np.concatenate((
            self.flat,
            np.full(missing, -1, dtype=self._sequence.dtype),
        ))
        array = flattened.reshape(N, T)
        return Sequence(array, name=self.name, start=self.start, end=self.end)

    def positions(
        self,
        start: int | None = None,
        end: int | None = None,
        /,
    ) -> "Sequence":
        """Returns a :class:`Sequence` object that only includes
        nucleotides from the specified range.

        Args:
            start (int, optional): First index of the specified range.
            end (int, optional): Last index of the specified range.
        """
        if end is None:
            if start is None:
                start, end = 0, self.size
            else:
                end = start
                start = 0
        if end > self.end:
            raise IndexError(
                f"Index {end} out of bounds for range "
                f"({self.start}, {self.end})"
            )
        act_start: int = start - self.start  # type: ignore
        act_end = end - self.start

        seq = Sequence(
            self.flat[np.newaxis, act_start:act_end],
            name=self.name,
            start=start,  # type: ignore
            end=end,
        )
        return seq.resample(self.T)

    def occurences(
        self,
        separate_repeat_masked: bool = False,
    ) -> dict[str, float]:
        """Counts the number of occurences of each nucleotide.

        Args:
            separate_repeat_masked (bool, optional): Whether to separate
                repeat-masked nucleotides from non-masked. Defaults to
                False.
        """
        if separate_repeat_masked:
            probs = {token: 0. for token in "ACGTNacgt"}
        else:
            probs = {token: 0. for token in "ACGTN" if token.upper() == token}
        size = self.size
        for index, token in enumerate("ACGTNacgt"):
            if not separate_repeat_masked:
                token = token.upper()
            probs[token] += (self.nuc == index).sum() / size
        return probs

    def __str__(self) -> str:
        return f"{self.name!r}[{self.start}:{self.end}]"

    def __repr__(self) -> str:
        return f"Sequence({self.name!r}, {self.start}, {self.end})"


class FASTA:
    """This class manages a sequence of nucleotides that is grouped into
    chunks of a given length with additional information about their
    origin (sequence name and range).

    It is likely that you never initialize this class by hand but
    instead use the corresponding loading method
    :meth:`bricks2marble.load_fasta`.

    Args:
        sequences (list[Sequence]): An list of :class:`Sequence`
            objects.
    """

    def __init__(self, sequences: list[Sequence]) -> None:
        self._sequences = sequences

    @property
    def nuc(self) -> np.ndarray:
        """Sequences of encoded nucleotides of shape ``(N, T)``."""
        return np.concatenate(
            [self._sequences[k].nuc for k in range(len(self._sequences))],
        )

    @property
    def size(self) -> int:
        return sum(seq.size for seq in self._sequences)

    @property
    def segments(self) -> list[Segment]:
        """Sequences of segments of length ``N``."""
        segs = []
        for k in range(len(self._sequences)):
            segs.extend(self._sequences[k].segments())
        return segs

    @property
    def N(self) -> int:
        return self.nuc.shape[0]

    @property
    def T(self) -> int:
        return self.nuc.shape[1]

    def resample(self, T: int) -> "FASTA":
        """Returns a :class:`FASTA` object that has the same sequences
        grouped into chunks of the given length. This can lead to
        differently padded sequences.
        """
        return FASTA([seq.resample(T) for seq in self._sequences])

    def one_hot(
        self,
        sequences: np.ndarray | None = None,
        pad_index: int = 4,
    ) -> np.ndarray:
        """Returns a one-hot encoded version of :meth:`FASTA.nuc` of
        shape ``(N, T, 6)``.

        Args:
            sequences (np.ndarray, optional): Which sequences to encode.
                Defaults to ``self.nuc``.
            pad_index (int, optional): What to replace the padding
                character (-1) by before encoding. Default to 4, which
                is an `N`.
        """
        nuc = self.nuc if sequences is None else sequences
        nuc[nuc == -1] = pad_index
        return ENCODING[nuc]

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
        occs = [
            seq.occurences(separate_repeat_masked) for seq in self._sequences
        ]
        return {
            k: sum(
                self._sequences[i].size * d[k] for i, d in enumerate(occs)
            ) / self.size
            for k in occs[0]
        }

    @overload
    def __getitem__(self, key: int) -> Sequence:
        ...

    @overload
    def __getitem__(self, key: int | str) -> Sequence | list[Sequence]:
        ...

    def __getitem__(self, key: int | str) -> Sequence | list[Sequence]:
        if isinstance(key, str):
            seqs = []
            for seq in self._sequences:
                if seq.name == key:
                    seqs.append(seq)
            if len(seqs) == 0:
                raise KeyError(f"No sequence named {key!r}")
            return seqs[0] if len(seqs) == 1 else seqs
        return self._sequences[key]

    def __str__(self) -> str:
        return "[" + ", ".join(str(seq) for seq in self._sequences) + "]"

    def __repr__(self) -> str:
        return "FASTA(" + ", ".join(str(seq) for seq in self._sequences) + ")"
