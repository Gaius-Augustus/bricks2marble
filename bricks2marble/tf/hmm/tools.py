import numpy as np
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


@tf.function
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


@tf.function
def left_right_3mers(nuc: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    left_3mers = tf.concat([
        make_kmer(
            nuc,
            k=3,
            pivot_left=True,
            collapse_pivot=True,
        ),
        tf.ones(
            tf.concat([tf.shape(nuc)[:-1], [1]], axis=0),  # type: ignore
            dtype=nuc.dtype,  # type: ignore
        ) / 64,
    ], axis=-1)
    right_3mers = tf.concat([
        make_kmer(
            nuc,
            k=3,
            pivot_left=False,
            collapse_pivot=True,
        ),
        tf.ones(
            tf.concat([tf.shape(nuc)[:-1], [1]], axis=0),  # type: ignore
            dtype=nuc.dtype,  # type: ignore
        ) / 64,
    ], axis=-1)

    return left_3mers, right_3mers  # type: ignore


def get_nuc_emission_distribution(
    start_codons: list[tuple[str, float]],
    stop_codons: list[tuple[str, float]],
    intron_begin_pattern: list[tuple[str, float]],
    intron_end_pattern: list[tuple[str, float]],
    intron_state_chain: int = 1,
    heads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates an emission probability matrix that imposes genetic
    rules on codons given codon distributions for different emerging
    patterns.

    The assumed order of states is
    ```
        (IR, I0, I1, I2, E0, E1), E2,
        START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    ```
    where the first 6 states do not have any codon restrictions.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays of shape
            ``(heads, 12+3*isc, 65)`` for left and right pivoted codons,
            where ``isc`` is the intron_state_chain argument.
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

    left_codon_probs = tf.concat(
        [left_codon_probs[0], tf.zeros((9, 1))],  # type: ignore
        axis=-1,
    )
    right_codon_probs = tf.concat(
        [right_codon_probs[0], tf.zeros((9, 1))],  # type: ignore
        axis=-1,
    )
    non_restricted_states = tf.concat([
        tf.zeros((3+3*intron_state_chain, 64)),
        tf.ones((3+3*intron_state_chain, 1)),
    ], axis=-1)
    left_codon_probs = tf.concat([
        non_restricted_states,
        left_codon_probs,
    ], axis=0)
    right_codon_probs = tf.concat([
        non_restricted_states,
        right_codon_probs,
    ], axis=0)

    left_codon_probs = tf.tile(
        tf.expand_dims(left_codon_probs, axis=0),
        (heads, 1, 1),
    )
    right_codon_probs = tf.tile(
        tf.expand_dims(right_codon_probs, axis=0),
        (heads, 1, 1),
    )

    return left_codon_probs.numpy(), right_codon_probs.numpy()  # type: ignore


def state_transitions(
    isc: int = 1,
    T_exon: int = 100,
    T_intron: int = 10000,
    T_ir: int = 10000,
    heads: int = 1,
) -> tuple[
    list[tuple[int, int, int]], np.ndarray, list[tuple[int, int]],
]:
    indices = np.array([
        # IR -> IR -> START -> E1 -> STOP -> IR
        [ 0,  0, np.log(T_ir - 1)],
        [ 0,  7, 0],
        [ 7,  5, 0],
        [ 5, 14, np.log(1/2)],
        [14,  0, 0],

        # E0 -> E1 -> E2 -> E0
        [ 4,  5, np.log(T_exon - 1)],
        [ 5,  6, np.log(T_exon - 1)],
        [ 6,  4, np.log(T_exon - 1)],

        # Ek -> EIk
        [ 4,  8, 0],
        [ 5,  9, np.log(1/2)],
        [ 6, 10, 0],

        # IEk -> Ek
        [11, 4, 0],
        [12, 5, 0],
        [13, 6, 0],
    ])
    # intron loops
    intron_loops = np.array([
        # loops on intron states Ikj -> Ikj
        [k, k] for k in range(1, 3*isc+1)
    ])
    intron_edges = np.array([
        # edges between intron states Ikj -> Ik(j+1)
        [k+j*3, k+(j+1)*3] for j in range(isc-1) for k in range(1, 4)
    ])
    intron_ingoing = np.array([
        # ingoing edges EIk -> Ik0
        [ 8+3*(isc-1), 1],
        [ 9+3*(isc-1), 2],
        [10+3*(isc-1), 3],
    ])
    intron_outgoing = np.array([
        # outgoing edges Ikj -> IEk
        [k+3*j, 10+k+3*(isc-1)] for j in range(isc) for k in range(1, 4)
    ])
    values = indices[:, 2].astype(np.float32)
    indices = indices[:, :2].astype(np.int64)
    if isc > 1:
        indices[indices > 0] = indices[indices > 0] + 3*(isc-1)
    n_edges = len(indices)
    indices = np.r_[
        indices,
        intron_loops,
        intron_edges,
        intron_ingoing,
        intron_outgoing,
    ]

    repeats = np.arange(heads).reshape(heads, 1, 1)
    repeats = np.tile(repeats, (1, len(indices), 1))
    indices = np.tile(indices, (heads, 1, 1))
    indices = np.concatenate([repeats, indices], axis=-1, dtype=np.int64)
    indices = indices.reshape(-1, 3)

    n_intron_loops = len(intron_loops)
    n_intron_edges = len(intron_edges)
    n_ingoing = len(intron_ingoing)
    share = np.array(
        # loops on intron states
        [
            [n_edges+k*3, n_edges+(k+1)*3]
            for k in range(isc)
        ]
        # edges between intron states
        + [
            [n_edges+n_intron_loops+k*3,
             n_edges+n_intron_loops+(k+1)*3]
            for k in range(isc-1)
        ]
        # ingoing edges
        + [
            [n_edges+n_intron_loops+n_intron_edges,
             n_edges+n_intron_loops+n_intron_edges+3]
        ]
        # outgoing edges
        + [
            [n_edges+n_intron_loops+n_intron_edges+n_ingoing+k*3,
             n_edges+n_intron_loops+n_intron_edges+n_ingoing+(k+1)*3]
            for k in range(isc)
        ]
    )

    values = np.r_[
        values,
        np.array(
            [
                # loops of intron states
                np.log(T_intron / isc - 1) + np.random.normal(scale=1e-2)
                for _ in range(isc)
            ]
            + [0] * (2 * isc)
        )
    ]

    values = np.exp(values) / np.sum(np.exp(values), -1, keepdims=True)
    values = np.tile(values, heads)

    # share = np.concatenate(
    #     [share + i*n_indices for i in range(heads)],
    #     axis=0,
    # )
    return indices.tolist(), values, share


def state_start_dist(
    isc: int = 1,
    heads: int = 1,
) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]]]:
    indices = np.array([
        [h, j] for h in range(heads) for j in range(12+3*isc)
    ])
    values = np.array([
        np.log(100),
        np.log(4),
        np.log(10), np.log(20), np.log(10),
        0,
    ], dtype=np.float32)
    values = np.exp(values) / np.sum(np.exp(values), -1, keepdims=True)
    values = np.tile(values, heads)

    share = np.array([
        [1, 1+3*isc], [4+3*isc,12+3*isc]
    ])
    share = np.concatenate(
        [share + i*(12+3*isc) for i in range(heads)],
        axis=0,
    )
    return indices.tolist(), values, share.tolist()


def state_names(isc: int = 1) -> list[str]:
    return (
        ["IR"]
        + [f"I{k}{j}" for j in range(isc) for k in range(3)]
        + ["E0", "E1", "E2"]
        + ["START", "EI0", "EI1", "EI2"]
        + ["IE0", "IE1", "IE2", "STOP"]
    )
