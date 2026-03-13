from collections.abc import Iterator
from typing import Callable, Literal, overload

import numpy as np
from pydantic import BaseModel


class Segment(BaseModel):
    """A segment has a name and a defined range. It is typically used to
    determine the origin of a sequence of nucleotides within a greater
    context, e.g. a genome.
    Counting starts at 0 and ends with T-1, like standard indexing in
    Python.
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
    Counting starts at 0 and ends with T-1, like standard indexing in
    Python.
    """

    strand: Literal["+", "-"] = "+"

    def __eq__(self, other) -> bool:
        if isinstance(other, Region):
            return super().__eq__(other)
        raise NotImplementedError("Can only compare Region to Region type")


def one_hot(
    sequences: np.ndarray,
    pad_index: int = 4,
    repeats: Literal["track", "expand", "omit"] = "track",
    N: Literal["track", "uniform"] = "track",
    dtype: type = np.float32,
) -> np.ndarray:
    """Returns a one-hot encoded version of :meth:`Sequence.nuc` of
    shape ``(N, T, D)``, where ``D`` can be 4, 5 or 6, depending on
    the options below.

    Args:
        sequences (np.ndarray, optional): Which sequences to encode.
            Defaults to ``self.nuc``.
        pad_index (int, optional): What to replace the padding
            character (-1) by before encoding. Default to 4, which
            is an `N`.
        repeats (str, optional): Changes the way repeat-masked
            positions are represented in the output. Can be either
            "track", "expand" or "omit". Defaults to "track".
            - "track": One additional dimension serves as a flag if
            the position is repeat-masked or not.
            - "expand": Four additional dimensions that are one-hot
            encodings of the four possible lower-case letters.
            - "omit": Do not include repeat information.
        N (str, optional): Changes the way N-tokens are encoded. Can
            be either "track" or "uniform". Defaults to "track".
            - "track": Adds another dimension to each position.
            - "uniform": Represents N as a uniform distribution over
                the four nucleotides.
        dtype (numpy dtype, optional): The dtype of the output encoding.
            Defaults to `float32`.
    """
    nuc = sequences.copy()
    nuc[nuc == -1] = pad_index

    match repeats:
        case "track" | "omit":
            encoding = np.array([
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
            if repeats == "omit":
                encoding = encoding[:, :5]
        case "expand":
            encoding = np.eye(9)
        case _:
            raise ValueError(f"Repeats mode {repeats!r} unknown.")

    encoded = encoding[nuc].astype(dtype)
    if N == "uniform":
        mask = encoded[..., 4] == 1
        encoded = np.concatenate([
            encoded[..., :4],
            encoded[..., 5:],
        ], axis=-1)
        encoded[mask, :4] = 1/4
    return encoded


def nucleotides_to_kmers(nuc: np.ndarray, k: int = 3) -> np.ndarray:
    nuc[nuc > 4] = nuc[nuc > 4] - 5
    nuc = np.asarray(nuc, dtype=np.uint16)
    nuc = nuc[:nuc.size//k*k].reshape(-1, k)
    shifts = 3 * np.arange(k-1, -1, -1, dtype=np.uint16)
    return np.bitwise_or.reduce(nuc << shifts, axis=1)


def complement(nuc: np.ndarray, reverse: bool = False) -> np.ndarray:
    if reverse: nuc = nuc[..., ::-1]
    nuc[nuc < 4] = 3 - nuc[nuc < 4]
    nuc[nuc > 4] = 13 - nuc[nuc > 4]
    return nuc


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
            ``Fasta.nuc``.
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
        self._evidence: np.ndarray | None = None

    @property
    def nuc(self) -> np.ndarray:
        return self._sequence

    @property
    def flat(self) -> np.ndarray:
        return self._sequence.flatten()[:self.size]

    @property
    def codons(self) -> np.ndarray:
        return nucleotides_to_kmers(self.flat)

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
                self._end = (self.N-1)*self.T + non_padded[0][0]
            else:
                self._end = self._sequence.size
            return self._end

    @property
    def evidence(self) -> np.ndarray | None:
        """Evidence is an extra type of information per base position.
        The given array has to have shape `(N, T, ...)`, matching the
        first two dimensions of `self.nuc`.
        """
        return self._evidence

    @evidence.setter
    def evidence(self, array: np.ndarray | None) -> None:
        if array is not None and self.nuc.shape != array.shape[:2]:
            raise ValueError(
                "Shape of given evidence does not match shape of sequence "
                "representation."
            )
        self._evidence = array

    def complement(self, reverse: bool = False) -> "Sequence":
        """Returns a new sequence object which is the complementary
        strand to the current sequence. This will not reverse the
        sequence direction unless specified.

        Args:
            reverse (bool, optional): Also reverses the sequence
                direction. For this, the sequence needs to be resampled
                to one chunk only. Defaults to False.
        """
        s = self.copy()
        if reverse: s.resample()
        s._sequence = complement(s._sequence, reverse=reverse)
        return s

    def is_repeat_masked(self) -> bool:
        """Checks if the sequence has any repeat-masked positions and
        returns a corresponding boolean.
        """
        return bool(np.any(self._sequence > 4))

    def segments(self) -> list[Segment]:
        return [
            Segment(
                name=self.name,
                start=self.start+(i-1)*self.T,
                end=min(self.start+i*self.T, self.start+self.size),
            ) for i in range(1, self.N+1)
        ]

    def flatten(self) -> "Sequence":
        """Sets the chunk size of this sequence to the total number of
        nucleotides without padding.

        This does not create a new sequence but overrides this one.
        """
        self.resample(self.size)
        return self

    def one_hot(
        self,
        pad_index: int = 4,
        repeats: Literal["track", "expand" , "omit"] = "track",
        N: Literal["track", "uniform"] = "track",
        dtype: type = np.float32,
    ) -> np.ndarray:
        """Returns a one-hot encoded version of :meth:`Sequence.nuc` of
        shape ``(N, T, D)``, where ``D`` can be 4, 5 or 6, depending on
        the options below.

        Args:
            pad_index (int, optional): What to replace the padding
                character (-1) by before encoding. Default to 4, which
                is an `N`.
            repeats (str, optional): Changes the way repeat-masked
                positions are represented in the output. Can be either
                "track", "expand" or "omit". Defaults to "track".
                - "track": One additional dimension serves as a flag if
                the position is repeat-masked or not.
                - "expand": Four additional dimensions that are one-hot
                encodings of the four possible lower-case letters.
                - "omit": Do not include repeat information.
            N (str, optional): Changes the way N-tokens are encoded. Can
                be either "track" or "uniform". Defaults to "track".
                - "track": Adds another dimension to each position.
                - "uniform": Represents N as a uniform distribution over
                    the four nucleotides.
            dtype (numpy dtype, optional): The dtype of the output
                encoding. Defaults to `float32`.
        """
        return one_hot(self.nuc, pad_index, repeats, N, dtype)

    def resample(
        self,
        T: int | None = None,
        drop_remainder: bool = False,
    ) -> "Sequence":
        """Resamples this sequence into chunks of the given length. This
        can lead to differently padded sequences.

        It does not create a new sequence but overrides this one.

        Args:
            T (int, optional): Length of the new chunks. If not
                specified, defaults to the total size of the sequence,
                meaning one chunk only.
            drop_remainder (bool, optional): If set to True, deletes the
                last chunk if the sequence has a length not divisable by
                ``T``. Defaults to False.
        """
        if T is None:
            T = self.size
        if T <= 0:
            raise ValueError(f"Unallowed chunk size: {T}")
        if drop_remainder:
            N = self.size // T
            self._end -= (self.size - N*T)
            self._sequence = self.flat[:N*T].reshape(N, T)
            if self.evidence is not None:
                if self.evidence is not None:
                    self.evidence = np.reshape(
                        self.evidence.flatten()[:self.size][:N*T],
                        (N, T),
                    )
            return self
        missing = (-self.size) % T
        N = (self.size + missing) // T
        flattened = np.concatenate((
            self.flat,
            np.full(missing, -1, dtype=self._sequence.dtype),
        ))
        array = flattened.reshape(N, T)
        self._sequence = array
        if self.evidence is not None:
            self.evidence = np.reshape(np.r_[
                self.evidence.flatten()[:self.size],
                np.full(missing, -1, dtype=self.evidence.dtype),
            ], (N, T))
        return self

    def positions(
        self,
        start: int | None = None,
        end: int | None = None,
        /,
    ) -> "Sequence":
        """Returns a :class:`Sequence` object that only includes
        nucleotides from the specified range. The sequence is resampled
        to one chunk only.

        Args:
            start (int, optional): First index of the specified range,
                starting at 0.
            end (int, optional): Last index of the specified range,
                itself excluded.
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
        if self.evidence is not None:
            seq.evidence = (self.evidence.flatten()[:self.size])[
                np.newaxis, act_start:act_end,
            ]
        return seq

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

    def string(self, repeats: bool = True) -> str:
        """Returns the string representation of this sequence as one
        long sequence of nucleotides.

        Args:
            repeats (bool, optional): Whether to write repeat-masked
                positions as lower-case letters. Defaults to True.
        """
        translation_table = bytes.maketrans(
            bytes([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            b"ACGTNacgt" if repeats else b"ACGTNACGT",
        )
        translated = self.flat.tobytes().translate(translation_table)
        return translated.decode("utf-8")

    def join(self, sequence: "Sequence") -> "Sequence":
        """Returns a new sequence that is the concatenation of the
        current and given sequence. The name will be the same as the
        current sequence. The start and end indices of the new sequence
        are set to the default values and might not correspond to the
        indices of the split sequences. The new sequence is resampled to
        the total length.
        """
        return Sequence(
            np.r_[self.flat, sequence.flat][np.newaxis, :],
            name=self.name,
        )

    def copy(self) -> "Sequence":
        return Sequence(
            self._sequence.copy(),
            name=self.name,
            start=self.start,
            end=self.end,
        )

    def __str__(self) -> str:
        return f"{self.name!r}[{self.start}:{self.end}]"

    def __repr__(self) -> str:
        return f"Sequence({self.name!r}, {self.start}, {self.end})"


class Fasta:
    """This class manages a sequence of nucleotides that is grouped into
    chunks of a given length with additional information about their
    origin (sequence name and range).

    It is likely that you never initialize this class by hand but
    instead use the corresponding loading method
    :meth:`bricks2marble.load_fasta`.

    Args:
        sequences (list[Sequence]): A list of :class:`Sequence` objects.
    """

    def __init__(self, sequences: list[Sequence]) -> None:
        self._sequences = sequences

    def is_repeat_masked(self) -> bool:
        """Checks if the Fasta has any repeat-masked positions and
        returns a corresponding boolean.
        """
        return any(seq.is_repeat_masked() for seq in self)

    @property
    def nuc(self) -> np.ndarray:
        """Sequences of encoded nucleotides of shape ``(N, T)``."""
        return np.concatenate([seq.nuc for seq in self])

    @property
    def evidence(self) -> np.ndarray | None:
        """Evidence of all sequences concatenated as an array of shape
        ``(N, T, ...)``. If one sequence does not contain evidence, the
        returned value is also None.
        """
        for seq in self:
            if seq.evidence is None: return None
        return np.concatenate([seq.evidence for seq in self])  # type: ignore

    @property
    def size(self) -> int:
        return sum(seq.size for seq in self._sequences)

    @property
    def segments(self) -> list[Segment]:
        """Sequences of segments of length ``N``."""
        segs = []
        for seq in self: segs.extend(seq.segments())
        return segs

    @property
    def N(self) -> int:
        return self.nuc.shape[0]

    @property
    def T(self) -> int:
        return self.nuc.shape[1]

    def complement(self, reverse: bool = False) -> "Fasta":
        """Returns a new fasta with all sequences being complemented.
        This will not reverse the sequence direction unless specified.

        Args:
            reverse (bool, optional): Also reverses the sequence
                direction. For this, all sequences need to be resampled
                to one chunk only. Defaults to False.
        """
        return Fasta([s.complement(reverse=reverse) for s in self])

    def resample(self, T: int, drop_remainder: bool = False) -> "Fasta":
        """Resamples the :class:`Fasta` object such that each sequence
        is grouped into chunks of the given length. This can lead to
        differently padded sequences.
        This method does not create a new Fasta object but is an
        in-place operation.

        Args:
            T (int): The target chunk size.
            drop_remainder (bool, optional): If set to True, deletes the
                last chunk in each sequence, if the sequence has a
                length not divisable by ``T``. Defaults to False.
        """
        for seq in self._sequences:
            seq.resample(T, drop_remainder=drop_remainder)
        return self

    def one_hot(
        self,
        pad_index: int = 4,
        repeats: Literal["track", "expand", "omit"] = "track",
        N: Literal["track", "uniform"] = "track",
        dtype: type = np.float32,
    ) -> np.ndarray:
        """Returns a one-hot encoded version of :meth:`Sequence.nuc` of
        shape ``(N, T, D)``, where ``D`` can be 4, 5 or 6, depending on
        the options below.

        Args:
            pad_index (int, optional): What to replace the padding
                character (-1) by before encoding. Default to 4, which
                is an `N`.
            repeats (str, optional): Changes the way repeat-masked
                positions are represented in the output. Can be either
                "track", "expand" or "omit". Defaults to "track".
                - "track": One additional dimension serves as a flag if
                the position is repeat-masked or not.
                - "expand": Four additional dimensions that are one-hot
                encodings of the four possible lower-case letters.
                - "omit": Do not include repeat information.
            N (str, optional): Changes the way N-tokens are encoded. Can
                be either "track" or "uniform". Defaults to "track".
                - "track": Adds another dimension to each position.
                - "uniform": Represents N as a uniform distribution over
                    the four nucleotides.
            dtype (numpy dtype, optional): The dtype of the output
                encoding. Defaults to `float32`.
        """
        return one_hot(self.nuc, pad_index, repeats, N, dtype)

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

    def rename(self, name: str | Callable[[str], str]) -> None:
        """Changes the name of all sequences in this Fasta object.

        Args:
            name (str | callable): If a string is given, changes the
                sequence names to that string. If a callable is given,
                this callable is applied to the sequence names, which
                are then set to the returned string.
        """
        if isinstance(name, str):
            rename = lambda _: name
        else:
            rename = name

        for seq in self._sequences:
            seq.name = rename(seq.name)

    def copy(self) -> "Fasta":
        return Fasta([seq.copy() for seq in self._sequences])

    @overload
    def __getitem__(self, key: int | str) -> Sequence:
        ...
    @overload
    def __getitem__(self, key: slice) -> "Fasta":
        ...
    def __getitem__(self, key: int | slice | str) -> "Sequence | Fasta":
        if isinstance(key, slice):
            return Fasta(self._sequences[key])
        if isinstance(key, int):
            return self._sequences[key]
        for seq in self._sequences:
            if seq.name == key:
                return seq
        raise KeyError(f"Sequence with name {key!r} does not exist.")

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self._sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __str__(self) -> str:
        return "[" + ", ".join(str(seq) for seq in self._sequences) + "]"

    def __repr__(self) -> str:
        return "Fasta(" + ", ".join(str(seq) for seq in self._sequences) + ")"
