from typing import Sequence

import tensorflow as tf


def is_codon_distribution(codons: list[tuple[str, float]]) -> bool:
    if sum(p for _, p in codons) != 1:
        return False

    for triplet, prob in codons:
        if len(triplet) != 3 or not (prob >= 0 and prob <= 1):
            return False

    return True


def encode_kmer(kmer: str, pivot_left=True) -> tf.Tensor:
    """Converts a k-mer to classes in the format ``(i,j)`` with
    ``i < 4**(k-1)`` and ``j < 4`` where ``4`` is the alphabet size.
    Example:
    ```
        if pivot_left:
            AAA -> (0,0), AAT -> (3,0), TAA -> (0,3)
        else:
            AAA -> (0,0), AAT -> (0,3), TAA -> (12, 0)
    ```
    """
    alphabet_with_unknown = "ACGTN"
    kmer_tensor = tf.constant([alphabet_with_unknown.index(x) for x in kmer])
    one_hot = tf.one_hot(kmer_tensor, len(alphabet_with_unknown))
    encoded_kmers = make_kmer(
        one_hot[tf.newaxis, ...],
        k=len(kmer),
        pivot_left=pivot_left,
    )
    if pivot_left:
        return tf.squeeze(encoded_kmers)[0]
    else:
        return tf.squeeze(encoded_kmers)[-1]


def make_codon_probs(
    codons: list[tuple[str, float]],
    pivot_left: bool,
) -> tf.Tensor:
    if not is_codon_distribution(codons):
        raise ValueError(
            "Given codon probabilities do not form a valid distribution"
        )

    codon_probs = sum(
        p * encode_kmer(triplet, pivot_left)  # type: ignore
        for triplet, p in codons
    )
    codon_probs = tf.reshape(codon_probs, [-1])
    return codon_probs[tf.newaxis, tf.newaxis, :]


def make_kmer(
    x: tf.Tensor,
    k: int,
    pivot_left: bool = True,
    collapse_pivot: bool = False,
) -> tf.Tensor:
    """Maps one-hot encoded nucleotide sequences to k-mer
    representations.

    Args:
        x (tf.Tensor): A tensor of shape ``(..., T, 5)`` representing
            sequences of length T. Assumes that the last dimension is
            one-hot encoded with "N" corresponding to the last position.
        k (int): The length of the k-mer.
        pivot_left (bool, optional): Whether to pivot the k-mer to the
            left or right.
        collapse_pivot (bool, optional): Whether to collapse the last
            two dimensions of the returned tensor into one. Defaults to
            False.

    Returns:
        tf.Tensor: A tensor of shape ``(..., T, 4**(k-1), 4)`` (or
            ``(..., T, 4**k)`` if `collapse_pivot`). If pivot_left is
            True, the last dimension corresponds to the 4 possible
            nucleotides in the leftmost position of the k-mer.
            Otherwise, the last dimension corresponds to the rightmost
            position in the k-mer. If the k-mer contains N, this is
            expressed equiprobably among the regular 4 nucleotides
            possible at that position.
    """
    L = tf.shape(x)[-2]  # type: ignore
    D = tf.cast(tf.shape(x)[-1] - 1, x.dtype)  # type: ignore

    base_probs = (
        x[..., :-1] + (tf.cast(x[..., -1:] == 1, x.dtype) / D)  # type: ignore
    )
    pad = tf.ones_like(base_probs[..., :k-1, :]) / D
    if pivot_left:
        padded = tf.concat([base_probs, pad], axis=-2)
        k_mer = padded[..., :L, tf.newaxis, :]  # type: ignore
    else:
        padded = tf.concat([pad, base_probs], axis=-2)
        k_mer = padded[..., k-1:L+k-1, tf.newaxis, :]  # type: ignore

    indices = range(1, k) if pivot_left else range(k-2, -1, -1)
    for i in indices:
        shift = padded[..., i:i+L, tf.newaxis, :, tf.newaxis]  # type: ignore
        k_mer = k_mer[..., tf.newaxis, :] * shift
        shape = [4**i, 4] if pivot_left else [4**(k-i-1), 4]
        k_mer = tf.reshape(
            k_mer,
            tf.concat([tf.shape(k_mer)[:-3], shape], axis=0),  # type: ignore
        )

    if collapse_pivot:
        return tf.reshape(
            k_mer,
            tf.concat((tf.shape(k_mer)[:-2], [4**k]), axis=0),  # type: ignore
        )
    return k_mer


def get_nuc_emission_distribution(
    start_codons: list[tuple[str, float]],
    stop_codons: list[tuple[str, float]],
    intron_begin_pattern: list[tuple[str, float]],
    intron_end_pattern: list[tuple[str, float]],
) -> tf.Tensor:
    """Generates an emission probability matrix that imposes genetic
    rules on codons given codon distributions for different emerging
    patterns.

    The assumed order of states is
    ```
        (IR, I0, I1, I2, E0, E1), E2,
        START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    ```
    where the first 6 states do not have any codon restrictions and are
    therefore omitted.

    Returns:
        tf.Tensor: A tensor of shape ``(2, 15, 64)`` for left and right
            pivoted codons.
    """
    start_codon_probs = make_codon_probs(start_codons, True)
    stop_codon_probs = make_codon_probs(stop_codons, False)
    intron_begin_codon_probs = make_codon_probs(intron_begin_pattern, True)
    intron_end_codon_probs = make_codon_probs(intron_end_pattern, False)
    any_codon_probs = make_codon_probs([("NNN", 1.)], False)

    not_stop_codon_probs = any_codon_probs * tf.cast(
        stop_codon_probs == 0,
        dtype=stop_codon_probs.dtype,
    )  # type: ignore
    not_stop_codon_probs /= tf.reduce_sum(not_stop_codon_probs)

    left_codon_probs = tf.concat(
        [any_codon_probs]
            + [start_codon_probs]
            + [intron_begin_codon_probs]*3
            + [any_codon_probs]*4,
        axis=1,
    )
    right_codon_probs = tf.concat(
        [not_stop_codon_probs]
            + [any_codon_probs]*2
            + [not_stop_codon_probs]
            + [any_codon_probs]
            + [intron_end_codon_probs]*3
            + [stop_codon_probs],
        axis=1,
    )

    return tf.concat(
        [left_codon_probs, right_codon_probs],
        axis=0,
    )  # type: ignore


def is_intergenic_loop(edge: Sequence[int]) -> bool:
    return edge[1]==edge[2] and edge[1] == 0


def is_intron_loop(edge: Sequence[int], k: int = 1) -> bool:
    return edge[1]==edge[2] and edge[1] > 0 and edge[1] < 1+3*k


def is_exon_transition(edge: Sequence[int], k: int = 1) -> bool:
    found_any = False
    exon_offset = 1+3*k
    for _ in range(k):
        found = (
            edge[2]-exon_offset == (edge[1]-exon_offset+k)%(3*k)
                and edge[1] >= exon_offset
                and edge[1] < exon_offset+3*k
        )
        found_any = found_any or found
    return found_any


def is_exon_1_out_transition(edge: Sequence[int], k: int = 1) -> bool:
    return edge[1] >= 1+4*k and edge[1] < 1+5*k and edge[1] != edge[2]


def is_intergenic_out_transition(edge: Sequence[int], k: int = 1) -> bool:
    return edge[1] == 0 and edge[2] != 0

