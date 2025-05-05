import numpy as np

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


class FASTA:

    def __init__(
        self,
        sequences: np.ndarray,
        choords: list[tuple[str, int, int]],
    ) -> None:
        self.sequences = sequences
        self.choords = choords
        self.repeat_masking = np.any(sequences > 4)

    @property
    def n_samples(self) -> int:
        return self.sequences.shape[0]

    @property
    def T(self) -> int:
        return self.sequences.shape[1]

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
        output_repeat_masked: bool = False,
    ) -> np.ndarray:
        if end is None:
            if start is None:
                start, end = 0, -1
            else:
                end = start
                start = 0
        sequences = self.sequences.reshape(-1)[start:end]
        if not output_repeat_masked:
            sequences[sequences > 4] = sequences[sequences > 4] - 5
        return sequences
