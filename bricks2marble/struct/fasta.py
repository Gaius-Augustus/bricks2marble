import numpy as np
from pydantic import BaseModel
from typing import Literal

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


class Coordinate(BaseModel):

    name: str
    start: int
    end: int


class Region(Coordinate):

    strand: Literal["+", "-"] = "+"


class FASTA:

    def __init__(
        self,
        sequences: np.ndarray,
        coords: list[Coordinate],
    ) -> None:
        self.sequences = sequences
        self.coords = coords
        self.repeat_masking = np.any(sequences > 4)

    @property
    def n_samples(self) -> int:
        return self.sequences.shape[0]

    @property
    def T(self) -> int:
        return self.sequences.shape[1]

    def resample(self, T: int, /, pad: int = -1) -> "FASTA":
        sequences = []
        new_coords = []

        file_data = {}
        for row, c in zip(self.sequences, self.coords):
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
                    chunk += [pad] * (T - len(chunk))

                sequences.append(chunk)
                new_coords.append(Coordinate(
                    name=seq,
                    start=chunk_start+1,
                    end=chunk_end,
                ))

        return FASTA(np.array(sequences), new_coords)

    def one_hot(self, sequences: np.ndarray | None = None) -> np.ndarray:
        seq = self.sequences if sequences is None else sequences
        seq[seq == -1] = 4
        if not self.repeat_masking:
            return ENCODING[:5, :5][seq]
        return ENCODING[seq]

    def occurences(
        self,
        separate_repeat_masked: bool = False,
    ) -> dict[str, float]:
        if self.repeat_masking and separate_repeat_masked:
            probs = {token: 0. for token in MAP.keys() if token != "n"}
        else:
            probs = {token: 0. for token in MAP.keys()
                     if token.upper() == token}
        size = (self.sequences[self.sequences != -1]).size
        if self.repeat_masking:
            for token, index in MAP.items():
                if token == "n": continue
                if not separate_repeat_masked:
                    token = token.upper()
                probs[token] += (self.sequences == index).sum() / size
        else:
            for token, index in MAP.items():
                if token.upper() == token:
                    probs[token] += (self.sequences == index).sum() / size
        return probs

    def positions(
        self,
        start: int | None = None,
        end: int | None = None,
        /,
        pad: int = -1,
    ) -> "FASTA":
        if end is None:
            if start is None:
                start, end = 0, -1
            else:
                end = start
                start = 0

        flat_numbers = []
        flat_positions = []

        global_index = 0
        for row, c in zip(self.sequences, self.coords):
            valid_len = c.end - c.start + 1
            for i in range(valid_len):
                if start <= global_index < end:  # type: ignore
                    flat_numbers.append(row[i])
                    flat_positions.append((c.name, start + i))  # type: ignore
                global_index += 1
            if global_index >= end:
                break

        grouped = {}
        for (seq, idx), val in zip(flat_positions, flat_numbers):
            if seq not in grouped:
                grouped[seq] = []
            grouped[seq].append((idx, val))

        new_seqs = []
        new_coords = []

        for seq, (positions, values) in grouped.items():
            total = len(values)
            num_chunks = (total + self.T - 1) // self.T

            for i in range(num_chunks):
                chunk_vals = list(values[i*self.T:(i+1)*self.T])
                chunk_pos = list(positions[i*self.T:(i+1)*self.T])

                if len(chunk_vals) < self.T:
                    chunk_vals += [-1] * (self.T - len(chunk_vals))

                new_seqs.append(chunk_vals)
                new_coords.append(
                    Coordinate(
                        name=seq,
                        start=chunk_pos[0] + 1,
                        end=chunk_pos[-1] + 1,
                    ) if chunk_pos else Coordinate(name=seq, start=-1, end=-1)
                )

        return FASTA(np.array(new_seqs), new_coords)
