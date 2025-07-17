from pathlib import Path

import numpy as np

import bricks2marble as b2m


def test_fasta(fasta_path: Path) -> None:
    fasta = b2m.io.load_fasta(fasta_path, T=5)

    assert fasta.nuc.shape == (9, 5)

    np.testing.assert_allclose(
        fasta.nuc,
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

    assert fasta.segments == [
        b2m.struct.Segment(name="name abc", start=0, end=5),
        ("name abc", 5, 10),
        ("name abc", 10, 15),
        ("name abc", 15, 20),
        ("name abc", 20, 25),
        ("name abc", 25, 27),
        ("name def", 0, 5),
        ("name def", 5, 10),
        ("name def", 10, 11),
    ]


def test_resample(fasta_path: Path) -> None:
    fasta = b2m.io.load_fasta(fasta_path, T=5)

    assert fasta.nuc.shape == (9, 5)

    fasta = fasta.resample(8)

    assert fasta.nuc.shape == (6, 8)

    np.testing.assert_allclose(
        fasta.nuc,
        np.array([
            [2, 0, 1, 1, 3, 2, 2, 0],
            [1, 4, 4, 1, 3, 3, 2, 5],
            [5, 7, 7, 0, 4, 4, 4, 1],
            [2, 0, 3, -1, -1, -1, -1, -1],
            [0, 0, 3, 8, 7, 7, 8, 0],
            [1, 4, 4, -1, -1, -1, -1, -1],
        ])
    )

    one_hot = fasta.one_hot()

    np.testing.assert_allclose(
        one_hot[2, :, :],
        np.array([
            [1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
        ])
    )

    np.testing.assert_allclose(
        one_hot[5, :, :],
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
        ])
    )

    assert fasta.segments == [
        b2m.struct.Segment(name="name abc", start=0, end=8),
        ("name abc", 8, 16),
        ("name abc", 16, 24),
        ("name abc", 24, 27),
        ("name def", 0, 8),
        ("name def", 8, 11),
    ]


def test_slice(fasta_path: Path) -> None:
    fasta = b2m.io.load_fasta(fasta_path, T=5)
    seq = fasta[0].positions(14)
    np.testing.assert_allclose(
        seq.nuc,
        np.array([
            [2, 0, 1, 1, 3],
            [2, 2, 0, 1, 4],
            [4, 1, 3, 3, -1],
        ])
    )
    assert seq.segments() == [
        ("name abc", 0, 5),
        ("name abc", 5, 10),
        ("name abc", 10, 14),
    ]

    fasta = b2m.io.load_fasta(fasta_path, T=5)
    seq = fasta[0].positions(27)
    np.testing.assert_allclose(
        seq.nuc,
        np.array([
            [2, 0, 1, 1, 3],
            [2, 2, 0, 1, 4],
            [4, 1, 3, 3, 2],
            [5, 5, 7, 7, 0],
            [4, 4, 4, 1, 2],
            [0, 3, -1, -1, -1],
        ])
    )
    assert seq.segments() == [
        ("name abc", 0, 5),
        ("name abc", 5, 10),
        ("name abc", 10, 15),
        ("name abc", 15, 20),
        ("name abc", 20, 25),
        ("name abc", 25, 27),
    ]

    fasta = b2m.io.load_fasta(fasta_path, T=5)
    seq = fasta[0].positions(7, 27)
    np.testing.assert_allclose(
        seq.nuc,
        np.array([
            [0, 1, 4, 4, 1],
            [3, 3, 2, 5, 5],
            [7, 7, 0, 4, 4],
            [4, 1, 2, 0, 3],
        ])
    )
    print(seq.segments())
    assert seq.segments() == [
        ("name abc", 7, 12),
        ("name abc", 12, 17),
        ("name abc", 17, 22),
        ("name abc", 22, 27),
    ]
