from pathlib import Path

import numpy as np

import bricks2marble as b2m

def test_fasta(fasta_path: Path) -> None:
    fasta = b2m.load_fasta(fasta_path, T=5)

    assert fasta.sequences.shape == (9, 5)

    np.testing.assert_allclose(
        fasta.sequences,
        np.array([
            [2, 0, 1, 1, 3],
            [2, 2, 0, 1, 4],
            [4, 1, 3, 3, 2],
            [5, 5, 7, 7, 0],
            [4, 4, 4, 1, 2],
            [0, 3, -1, -1, -1],
            [0, 0, 3, 8, 7],
            [7, 8, 0, 1, 4],
            [4, -1, -1, -1, -1],
        ])
    )

    one_hot = fasta.one_hot()
    np.testing.assert_allclose(
        one_hot[3, :, :],
        np.array([
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
        ])
    )

    np.testing.assert_allclose(
        one_hot[5, :, :],
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
        ])
    )

    np.testing.assert_allclose(
        one_hot[6, :, :],
        np.array([
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1],
        ])
    )

    assert fasta.coords == [
        ("name abc", 1, 5),
        ("name abc", 6, 10),
        ("name abc", 11, 15),
        ("name abc", 16, 20),
        ("name abc", 21, 25),
        ("name abc", 26, 27),
        ("name def", 1, 5),
        ("name def", 6, 10),
        ("name def", 11, 11),
    ]
