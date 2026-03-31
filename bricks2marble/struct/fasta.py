from collections.abc import Iterator
from typing import Callable, Literal, overload

import numpy as np


def one_hot(
    sequences: np.ndarray,
    pad_index: int = 4,
    repeats: Literal["track", "expand", "omit"] = "track",
    N: Literal["track", "uniform"] = "track",
    dtype: type = np.float32,
) -> np.ndarray:
    """Returns a one-hot encoded version of an array of nucleotides of
    shape ``(N, T)`` as an array of shape ``(N, T, D)``, where ``D`` can
    be 4, 5 or 6, depending on the options below.

    Args:
        sequences (np.ndarray): Sequences to encode.
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
        sequences (list[Sequence]): An array of shape ``(L, )`` which
            holds encoded nucleotides.
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
        self._end = end if end > 0 else sequence.size
        self._T = None
        self._drop_remainder = False
        if sequence.ndim != 1:
            raise ValueError("Only 1D nucleotide sequences are supported")
        if self.size != sequence.size:
            raise ValueError(
                "Given boundary sequence indices do not match number of"
                f" nucleotides ({self._end}-{self._start} != {sequence.size})"
            )
        self._evidence: np.ndarray | None = None

    @property
    def nuc(self) -> np.ndarray:
        """Array of encoded nucleotides of shape ``(N, T)``."""
        return self._realize()

    @property
    def flat(self) -> np.ndarray:
        """Flat sequence of encoded nucleotides."""
        return self._sequence

    @property
    def codons(self) -> np.ndarray:
        """Flat sequence of encoded codons."""
        return nucleotides_to_kmers(self.flat.copy())

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def N(self) -> int:
        if self._T is None: return 1
        if self._drop_remainder: return self.size // self._T
        missing = (-self.size) % self._T
        return (self.size + missing) // self._T

    @property
    def T(self) -> int:
        return self.size if self._T is None else self._T

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def evidence(self) -> np.ndarray | None:
        """Evidence is an extra type of information per base position.
        The given array has to have shape ``(self.size, ...)``.
        The returned array `self.evidence` will have shape ``(N, T)``.
        """
        return self._realize_evidence()

    @evidence.setter
    def evidence(self, array: np.ndarray | None) -> None:
        if array is not None and self.size != array.shape[0]:
            raise ValueError(
                f"First axis of given evidence ({array.shape[0]}) does not "
                f"match size of nucleotide representation ({self.size})"
            )
        self._evidence = array

    def _realize(self) -> np.ndarray:
        T = self._T
        if T is None: return self.flat[np.newaxis, :]

        if self._drop_remainder:
            N = self.size // T
            self._end -= (self.size - N*T)
            return self.flat[:N*T].reshape(N, T)

        missing = (-self.size) % T
        N = (self.size + missing) // T
        return np.reshape(np.concatenate((
            self.flat,
            np.full(missing, -1, dtype=self.flat.dtype),
        )), (N, T))

    def _realize_evidence(self) -> np.ndarray | None:
        if self._evidence is None: return None

        T = self._T
        if T is None:
            T = self.size
        if T <= 0:
            raise ValueError(f"Unallowed chunk size: {T}")
        if self._drop_remainder:
            N = self.size // T
            self._end -= (self.size - N*T)
            return np.reshape(self._evidence[:N*T], (N, T))
        missing = (-self.size) % T
        N = (self.size + missing) // T
        return np.reshape(np.r_[
            self._evidence,
            np.full(missing, -1, dtype=self._evidence.dtype),
        ], (N, T))

    def complement(self, reverse: bool = False) -> "Sequence":
        """Returns a new sequence object which is the complementary
        strand to the current sequence. This will not reverse the
        sequence direction unless specified.

        Args:
            reverse (bool, optional): Also reverses the sequence
                direction. Defaults to False.
        """
        s = self.copy()
        s._sequence = complement(s._sequence, reverse=reverse)
        return s

    def is_repeat_masked(self) -> bool:
        """Checks if the sequence has any repeat-masked positions and
        returns a corresponding boolean.
        """
        return bool(np.any(self.flat > 4))

    def flatten(self) -> "Sequence":
        """Sets the chunk size of this sequence to the total number of
        nucleotides without padding.

        This does not create a new sequence but overrides this one.
        """
        return self.resample(None)

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
        """Resamples this sequence in-place into chunks of the given
        length. This does not actually change the internal
        representation. The actual reshaping happens once `self.nuc` is
        retrieved.

        Args:
            T (int, optional): Length of the new chunks. If not
                specified, defaults to the total size of the sequence,
                meaning one chunk only.
            drop_remainder (bool, optional): If set to True, deletes the
                last chunk if the sequence has a length not divisable by
                ``T``. Defaults to False.
        """
        self._T = T
        self._drop_remainder = drop_remainder
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
            self.flat[act_start:act_end],
            name=self.name,
            start=start,  # type: ignore
            end=end,
        )
        if self._evidence is not None:
            seq.evidence = self._evidence[act_start:act_end]
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
            probs[token] += (self.flat == index).sum() / size
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
        return Sequence(np.r_[self.flat, sequence.flat], name=self.name)

    def copy(self) -> "Sequence":
        seq = Sequence(
            self.flat.copy(),
            name=self.name,
            start=self.start,
            end=self.end,
        )
        return seq.resample(self._T, drop_remainder=self._drop_remainder)

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
    :meth:`bricks2marble.io.load_fasta`.

    Args:
        sequences (list[Sequence]): A list of :class:`Sequence` objects.
    """

    def __init__(self, sequences: list[Sequence]) -> None:
        self._sequences = sequences

    @property
    def nuc(self) -> np.ndarray:
        """Sequences of encoded nucleotides of shape ``(N, T)``."""
        self.T  # check for sequence ambiguity
        return np.concatenate([seq.nuc for seq in self])

    @property
    def evidence(self) -> np.ndarray | None:
        """Evidence of all sequences concatenated as an array of shape
        ``(N, T, ...)``. If one sequence does not contain evidence, the
        returned value is also None.
        """
        self.T  # check for sequence ambiguity
        ev = [s.evidence for s in self]
        ev_none = [e is None for e in ev]
        if any(ev_none) and not all(ev_none):
            raise RuntimeError("Ambiguous evidence across sequences")
        return np.concatenate(ev)  # type: ignore

    @property
    def size(self) -> int:
        return sum(s.size for s in self)

    @property
    def N(self) -> int:
        return sum(s.N for s in self)

    @property
    def T(self) -> int :
        Ts = {s.T for s in self}
        if len(Ts) > 1: raise RuntimeError(
            "Ambiguous chunk length across sequences, call Fasta.resample"
        )
        return Ts.pop()

    def is_repeat_masked(self) -> bool:
        """Checks if the Fasta has any repeat-masked positions and
        returns a corresponding boolean.
        """
        return any(s.is_repeat_masked() for s in self)

    def complement(self, reverse: bool = False) -> "Fasta":
        """Returns a new fasta with all sequences being complemented.
        This will not reverse the sequence direction unless specified.

        Args:
            reverse (bool, optional): Also reverses the sequence
                direction. For this, all sequences need to be resampled
                to one chunk only. Defaults to False.
        """
        return Fasta([s.complement(reverse=reverse) for s in self])

    def resample(
        self,
        T: int | None = None,
        drop_remainder: bool = False,
    ) -> "Fasta":
        """Resamples the :class:`Fasta` object in-place such that each
        sequence is grouped into chunks of the given length.

        Args:
            T (int): The target chunk size.
            drop_remainder (bool, optional): If set to True, deletes the
                last chunk in each sequence, if the sequence has a
                length not divisable by ``T``. Defaults to False.
        """
        for s in self: s.resample(T, drop_remainder=drop_remainder)
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
        occs = [seq.occurences(separate_repeat_masked) for seq in self]
        return {
            k: sum(self[i].size * d[k] for i, d in enumerate(occs)) / self.size
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
        return Fasta([seq.copy() for seq in self])

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
        return "[" + ", ".join(str(seq) for seq in self) + "]"

    def __repr__(self) -> str:
        return "Fasta(" + ", ".join(str(seq) for seq in self) + ")"
