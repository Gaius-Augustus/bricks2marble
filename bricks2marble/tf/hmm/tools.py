import numpy as np
import tensorflow as tf
from hidten.tf.util import safe_log


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
    uniform_N: bool = False,
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
        uniform_N (bool, optional): Whether the "N" is already encoded
            as a uniform distribution over the four nucleotides.
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
    D = tf.cast(tf.shape(x)[-1], x.dtype)  # type: ignore

    if uniform_N:
        base_probs = tf.identity(x)
    else:
        D -= 1
        base_probs = x[..., :-1] + (tf.cast(x[..., -1:] == 1, x.dtype) / D)
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


def left_right_3mers(
    nuc: tf.Tensor,
    uniform_N: bool = False,
    repeats: tf.Tensor | None = None,
    left_factor: tf.Tensor | None = None,
    right_factor: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    left_3mers = tf.concat([
        make_kmer(
            nuc,
            k=3,
            uniform_N=uniform_N,
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
            uniform_N=uniform_N,
            pivot_left=False,
            collapse_pivot=True,
        ),
        tf.ones(
            tf.concat([tf.shape(nuc)[:-1], [1]], axis=0),  # type: ignore
            dtype=nuc.dtype,  # type: ignore
        ) / 64,
    ], axis=-1)

    if repeats is not None:
        repeats = tf.concat((repeats, tf.reverse(repeats, [1])), 0)
    if left_factor is not None:
        left_3mers = left_3mers * (1.0 + repeats * (left_factor - 1.0))
    if right_factor is not None:
        right_3mers = right_3mers * (1.0 + repeats * (right_factor - 1.0))

    return left_3mers, right_3mers  # type: ignore


def get_repeats_emission_distribution(
    f: float,
    intron_state_chain: int = 1,
    heads: int = 1,
    spliced_stop: bool = False,
) -> np.ndarray:
    """Generates an emission probability matrix for encoding repeat
    information that downscales probabilities of coding regions within
    repeats.

    Returns:
        np.ndarray: Array of shape ``(heads, 12+3*isc, 2)``, where
            ``isc`` is the intron_state_chain argument.
    """
    if spliced_stop:
        emission = np.r_[
            np.ones((1+6*intron_state_chain), dtype=np.float32),
            np.full((17, ), fill_value=f, dtype=np.float32),
        ]
    else:
        emission = np.r_[
            np.ones((1+3*intron_state_chain), dtype=np.float32),
            np.full((11, ), fill_value=f, dtype=np.float32),
        ]
    emission = np.c_[
        np.ones_like(emission, dtype=np.float32),
        emission,
    ]
    emission = np.repeat(emission[np.newaxis, ...], repeats=heads, axis=0)
    return emission

def has_default_stop_codons(codons):
    default_stops = {"TAG", "TAA", "TGA"}
    return {codon for codon, _ in codons} == default_stops

def get_stop_codon_splice_sites(
    stop_codons: list[tuple[str, float]],
) -> list[tf.Tensor]:

    if len(stop_codons) != 3:
        raise ValueError(
            "non-default stop-codons cannot be forbidden"
            " at splice sites."
        )
    if not has_default_stop_codons(stop_codons):
        raise ValueError(
            "non-default stop-codons cannot be forbidden"
            " at splice sites."
        )
    stop_codon_probs = make_codon_probs(stop_codons, False)
    any_codon_probs = make_codon_probs([("NNN", 1.)], False)

    not_stop_codon_probs = any_codon_probs * tf.cast(
        stop_codon_probs == 0,
        dtype=stop_codon_probs.dtype,
    )  # type: ignore
    not_stop_codon_probs /= tf.reduce_sum(not_stop_codon_probs)

    TA_probs = make_codon_probs([("NTA", 1.)], False) 
    TG_probs = make_codon_probs([("NTG", 1.)], False)
    T_probs = make_codon_probs([("NNT", 1.)], False)
    A_probs = make_codon_probs([("ANN", 1.)], True)
    AG_probs = make_codon_probs([("ANN", 0.5),("GNN", 0.5)], True)
    AAAGGA_probs = make_codon_probs([("AAN", 0.34),("GAN", 0.33),("AGN", 0.33)], True)

    not_TATG_probs = any_codon_probs * tf.cast(
        TA_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    TA_probs = any_codon_probs * tf.cast(
        TA_probs > 0,
        dtype=stop_codon_probs.dtype,
    )
    not_TATG_probs = not_TATG_probs * tf.cast(
        TG_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    TG_probs = any_codon_probs * tf.cast(
        TG_probs > 0,
        dtype=stop_codon_probs.dtype,
    )
    not_T_probs = any_codon_probs * tf.cast(
        T_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    T_probs = any_codon_probs * tf.cast(
        T_probs > 0,
        dtype=stop_codon_probs.dtype,
    )
    not_A_probs = any_codon_probs * tf.cast(
        A_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    not_AG_probs = any_codon_probs * tf.cast(
        AG_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    not_AAAGGA_probs = any_codon_probs * tf.cast(
        AAAGGA_probs == 0,
        dtype=stop_codon_probs.dtype,
    )
    end_patterns = [any_codon_probs]*3 + [not_AAAGGA_probs] + [not_AG_probs] + [not_A_probs]
    begin_patterns = [not_TATG_probs] + [not_stop_codon_probs] + [not_T_probs] + [T_probs] + [TA_probs] + [TG_probs]

    return end_patterns, begin_patterns

def get_nuc_emission_distribution(
    start_codons: list[tuple[str, float]],
    stop_codons: list[tuple[str, float]],
    intron_begin_pattern: list[tuple[str, float]],
    intron_end_pattern: list[tuple[str, float]],
    intron_state_chain: int = 1,
    heads: int = 1,
    spliced_stop: bool = False,
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
    
    if not spliced_stop:
        add_states = 0
    else:
        add_states = 3

    if not spliced_stop:
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

    else:
        end_patterns, begin_patterns = get_stop_codon_splice_sites(stop_codons)
        left_codon_probs = tf.concat(
            [any_codon_probs]
                + [start_codon_probs]
                + [intron_begin_codon_probs]*6
                + end_patterns
                + [any_codon_probs],
            axis=1,
        )
        right_codon_probs = tf.concat(
            [not_stop_codon_probs]
                + [any_codon_probs]
                + begin_patterns
                + [intron_end_codon_probs]*6
                + [stop_codon_probs],
            axis=1,
        )

    left_codon_probs = tf.concat(
        [left_codon_probs[0], tf.zeros((9+2*add_states, 1))],  # type: ignore
        axis=-1,
    )
    right_codon_probs = tf.concat(
        [right_codon_probs[0], tf.zeros((9+2*add_states, 1))],  # type: ignore
        axis=-1,
    )

    non_restricted_states = tf.concat([
        tf.zeros((3+(3+add_states)*intron_state_chain, 64)),
        tf.ones((3+(3+add_states)*intron_state_chain, 1)),
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


def get_repeats_at_borders_multiplier(
    f: float,
    start_codons: list[tuple[str, float]],
    stop_codons: list[tuple[str, float]],
    intron_begin_pattern: list[tuple[str, float]],
    intron_end_pattern: list[tuple[str, float]],
) -> tuple[tf.Tensor, tf.Tensor]:
    left_repeats_factor = f * (
        tf.cast(make_codon_probs(start_codons, True) > 0, tf.float32)
        + tf.cast(make_codon_probs(intron_begin_pattern, True) > 0, tf.float32)
    )
    right_repeats_factor = f * (
        tf.cast(make_codon_probs(stop_codons, False) > 0, tf.float32)
        + tf.cast(make_codon_probs(intron_end_pattern, False) > 0, tf.float32)
    )
    left_repeats_factor = (
        left_repeats_factor + tf.cast(left_repeats_factor == 0, tf.float32)
    )
    right_repeats_factor = (
        right_repeats_factor + tf.cast(right_repeats_factor == 0, tf.float32)
    )
    left_repeats_factor = tf.concat(
        (left_repeats_factor, tf.ones((1, 1, 1))),
        axis=-1,
    )
    right_repeats_factor = tf.concat(
        (right_repeats_factor, tf.ones((1, 1, 1))),
        axis=-1,
    )
    return left_repeats_factor, right_repeats_factor


def emission_parameters(
    D: int,
    S: int,
    H: int,
    isc: int = 1,
    share_noncoding: bool = False,
    share_frames: bool = True,
    spliced_stop: bool = False,
) -> tuple[
    np.ndarray | tf.keras.Initializer,
    list[tuple[int, int, int]],
    list[tuple[int, int]] | None
]:
    allow = [
        (h, i, k)
        for h, states in enumerate([S]*H)
        for k in range(D)
        for i in range(states)
    ]

    if spliced_stop:
        intron_number = 6
        add_states = 3
    else:
        intron_number = 3
        add_states = 0

    share = []
    if share_noncoding:
        share += [
            (h*D*S+i*S, h*D*S+i*S+1+isc*intron_number)
            for h in range(H)
            for i in range(D)
        ]
    elif share_frames:
        share += [
            (h*D*S+i*S+1+j*intron_number, h*D*S+i*S+4+j*intron_number)
            for h in range(H)
            for i in range(D)
            for j in range(isc)
        ]

    if share_frames:
        share += (
            [
                (h*D*S+i*S+5+intron_number*isc, h*D*S+i*S+8+add_states+intron_number*isc)
                for h in range(H)
                for i in range(D)
            ] + [
                (h*D*S+i*S+8+add_states+intron_number*isc, h*D*S+i*S+11+(add_states*2)+intron_number*isc)
                for h in range(H)
                for i in range(D)
            ]
        )

    if len(share) == 0:
        share = None

    return tf.keras.initializers.GlorotNormal(), allow, share


def emission_parameters_eye(
    D: int,
    S: int,
    H: int,
    epsilon: float = 0,
) -> tuple[
    np.ndarray | tf.keras.Initializer,
    list[tuple[int, int, int]],
    list[tuple[int, int]] | None
]:
    allow = [
        (h, i, k)
        for h, states in enumerate([S]*H)
        for k in range(D)
        for i in range(states)
    ]

    values = np.eye(S, dtype=np.float32)
    values += epsilon / (S-1)
    values[np.diag_indices(S)] -= epsilon * (1 + 1 / (S-1))
    values = values.flatten()
    values = safe_log(values)
    if D != S: values = np.tile(values, D // S)
    return tf.keras.initializers.Constant(values), allow, None

def state_transitions_simple(
    share_noncoding: bool = False,
    share_frames: bool = True,
    p_IR: int | float | None = None,
    p_intron: int | float | None = None,
    p_exon: int | float | None = None,
    heads: int = 1,
    spliced_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Base case of `state_transitions` with no intron chains.
    Implementation is purely for better management and quicker testing
    of new features.
    """
    if p_IR is None: p_IR = 0.9999
    if p_exon is None: p_exon = 0.975
    if p_intron is None: p_intron = 0.99

    if p_IR < 1: p_IR = 1 / (1-p_IR)
    if p_exon < 1: p_exon = 1 / (1-p_exon)
    if p_intron < 1: p_intron = 1 / (1-p_intron)
    
    if spliced_stop: 
        add = 3
    else:
        add = 0

    indices = np.array([
        # IR -> IR
        [ 0,  0, np.log(p_IR - 1)],

        # Ik -> Ik
        [ 1,  1, np.log(p_intron - 1)],
        [ 2,  2, np.log(p_intron - 1)],
        [ 3,  3, np.log(p_intron - 1)],
    ])
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            [ 4,  4, np.log(p_intron - 1)],
            [ 5,  5, np.log(p_intron - 1)],
            [ 6,  6, np.log(p_intron - 1)],
        ])))
    indices = np.concatenate((indices, np.array([
        # IR -> START
        [ 0,  7+add, 0],

        # Ik -> IEk
        [ 1,  11+add*2, 0],
        [ 2,  12+add*2, 0],
        [ 3,  13+add*2, 0],
    ])))
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            [ 4,  20, 0],
            [ 5,  21, 0],
            [ 6,  22, 0],
        ])))
    indices = np.concatenate((indices, np.array([
        # EIk -> Ik
        [ 8+add,  1, 0],
        [ 9+add,  2, 0],
        [10+add,  3, 0],
    ])))
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            [14,  4, 0],
            [15,  5, 0],
            [16,  6, 0],
        ])))
    indices = np.concatenate((indices, np.array([
        # START -> E1 -> STOP -> IR
        [ 7+add,  5+add, 0],
        [ 5+add, 14+3*add, np.log(1/2)],
        [14+3*add,  0, 0],

        # E0 -> E1 -> E2 -> E0
        [ 4+add,  5+add, np.log(p_exon - 1)],
        [ 5+add,  6+add, np.log(p_exon - 1)],
        [ 6+add,  4+add, np.log(p_exon - 1)],

        # Ek -> EIk
        [ 4+add,  8+add, 0],
        [ 5+add,  9+add, np.log(1/2)],
        [ 6+add, 10+add, 0],
    ])))
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            [ 7,  15, 0],
            [ 7,  16, 0],
            [ 9, 14, 0],
        ])))
    indices = np.concatenate((indices, np.array([
        # IEk -> Ek
        [11+2*add, 4+add, 0],
        [12+2*add, 5+add, 0],
        [13+2*add, 6+add, 0],
    ])))
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            [21, 7, 0],
            [22, 7, 0],
            [20, 9, 0],
        ])))
    indices = indices[:, :2].astype(np.int64)

    share = []
    values = []
    if share_noncoding:
        share += [[0, 4+add], [4+add, 8+2*add]]
        values += [np.log(p_IR - 1), 0]
        values += 3 * [0]
        if spliced_stop:
            values += 3 * [0]
    elif share_frames:
        share += [[1, 4+add]]
        share += [[5+add, 8+2*add]]
        share += [[8+2*add, 11+3*add]]
        values += [np.log(p_IR - 1), np.log(p_intron - 1), 0, 0, 0]
    else:
        share = None
        values += [np.log(p_IR - 1)]
        values += 3 * [np.log(p_intron - 1)]
        if spliced_stop:
            values += 3 * [np.log(p_intron - 1)]
            values += 6 * [0]
        values += 4 * [0]
        values += 3 * [0]

    values += [0, np.log(1/2), 0]
    values += 3 * [np.log(p_exon - 1)]
    if not spliced_stop:
        values += [0, np.log(1/2), 0]
    else:
        values += [0, 0, 0, np.log(1/2), 0, 0]
        values += 3 * [0]
    values += 3 * [0]

    repeats = np.arange(heads).reshape(heads, 1, 1)
    repeats = np.tile(repeats, (1, len(indices), 1))
    indices = np.tile(indices, (heads, 1, 1))
    indices = np.concatenate([repeats, indices], axis=-1, dtype=np.int64)
    indices = indices.reshape(-1, 3)

    values = np.array(values).astype(np.float32)
    values = np.exp(values) / np.sum(np.exp(values), -1, keepdims=True)
    values = np.tile(values, heads)

    if share is not None:
        share = np.array(share)
        share = np.r_[*[
            share + i*(len(indices)//heads) for i in range(heads)
        ]]

    return indices, values, share

def state_transitions_ir(
    p_IR: int | float | None = None,
    spliced_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if spliced_stop:
        add = 3
    else:
        add = 0
    indices = np.array([
        [ 0,  7+add, 0],
        [14+3*add,  0, 0],
        [ 0,  0, np.log(p_IR - 1)],
    ])
    values = indices[:, 2].astype(np.float32)
    indices = indices[:, :2].astype(np.int64)
    return indices, values.tolist()

def state_transitions_introns(
    isc: int = 1,
    intron_chain_starts: bool = False,
    intron_chain_skips: bool = False,
    intron_chain_loop: bool = False,
    intron_chain_transitions: bool = True,
    p_intron: int | float | list[float | int] | None = None,
    heads: int = 1,
    share_frames: bool = True,
    spliced_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if spliced_stop:
        intron_number = 6
        add = 3
    else:
        intron_number = 3
        add = 0
    # intron loops
    intron_loops = np.array([
        # loops on intron states Ikj -> Ikj
        [k, k] for k in range(0, intron_number*isc)
    ])
    intron_edges = (np.array([
        # edges between intron states Ikj -> Ik(j+1)
        [k+j*intron_number, k+(j+1)*intron_number] for j in range(isc-1) for k in range(0, intron_number)
    ]) if intron_chain_transitions else np.array([]))
    intron_ingoing = np.array([
        # ingoing edges EIk -> Ik0
        [ 7+add+intron_number*(isc-1), 0],
        [ 8+add+intron_number*(isc-1), 1],
        [ 9+add+intron_number*(isc-1), 2],
    ])
    if spliced_stop:
        add_ingoing = np.array([
            # ingoing edges EIk -> Ik0
            [ 13+6*(isc-1), 3],
            [ 14+6*(isc-1), 4],
            [ 15+6*(isc-1), 5],
        ])
        intron_ingoing = np.concatenate((intron_ingoing, add_ingoing))
    if intron_chain_starts:
        for k in range(1, isc):
            intron_ingoing = np.r_[
                intron_ingoing,
                np.array([
                    # ingoing edges EIk -> Ikj
                    [ 7+add+intron_number*(isc-1), k*intron_number],
                    [ 8+add+intron_number*(isc-1), 1+k*intron_number],
                    [ 9+add+intron_number*(isc-1), 2+k*intron_number],
                ])
            ]
            if spliced_stop:
                for k in range(1, isc):
                    intron_ingoing = np.r_[
                        intron_ingoing,
                        np.array([
                            # ingoing edges EIk -> Ikj
                            [ 13+6*(isc-1), 3+k*intron_number],
                            [ 14+6*(isc-1), 4+k*intron_number],
                            [ 15+6*(isc-1), 5+k*intron_number],
                        ])
                ]
    if intron_chain_skips:
        intron_outgoing = np.array([
            # outgoing edges Ikj -> IEk
            [k+intron_number*j, 10+2*add+k+intron_number*(isc-1)] for j in range(isc) for k in range(0, intron_number)
        ])
            
    else:
        intron_outgoing = np.array([
            # outgoing edges Ik(-1) -> IEk
            [k+intron_number*(isc-1), 10+2*add+k+intron_number*(isc-1)] for k in range(0, intron_number)
        ])

    if intron_chain_loop:
        intron_repeat = np.array([
            [k+intron_number*(isc-1), k] for k in range(0, intron_number)
        ])
    
    if isc > 1:
        indices = intron_loops
        if intron_chain_transitions:
            indices = np.r_[indices, intron_edges]
        indices = np.r_[indices, intron_ingoing, intron_outgoing]
    else:
        indices = np.r_[
            intron_loops,
            intron_ingoing,
            intron_outgoing,
        ]
    if intron_chain_loop:
        indices = np.r_[indices, intron_repeat]

    n_intron_loops = len(intron_loops)
    n_intron_edges = len(intron_edges)
    n_ingoing = len(intron_ingoing)
    n_outgoing = len(intron_outgoing)
    share = np.array(
        # loops on intron states
        [
            [k*intron_number, (k+1)*intron_number]
            for k in range(isc)
        ]
        # edges between intron states
        + ([
            [n_intron_loops+k*intron_number,
             n_intron_loops+(k+1)*intron_number]
            for k in range(isc-1)
        ] if intron_chain_transitions else [])
        # ingoing edges
        + [
            [n_intron_loops+n_intron_edges+k*intron_number,
             n_intron_loops+n_intron_edges+(k+1)*intron_number]
            for k in range(isc if intron_chain_starts else 1)
        ]
        # outgoing edges
        + [
            [n_intron_loops+n_intron_edges+n_ingoing+k*intron_number,
             n_intron_loops+n_intron_edges+n_ingoing+(k+1)*intron_number]
            for k in range(isc if intron_chain_skips else 1)
        ]
        # chain loop edges
        + ([
            [n_intron_loops+n_intron_edges+n_ingoing+n_outgoing,
             n_intron_loops+n_intron_edges+n_ingoing+n_outgoing+intron_number]
        ] if intron_chain_loop else [])
    )

    intron_loop_values = [np.log(pi - 1) for pi in p_intron]
    values = np.array(
            # loops on intron states
            intron_loop_values
            # edges between intron states
            + (([np.log(1/2) if intron_chain_skips else 0] * (isc-1))
               if intron_chain_transitions else [])
            # ingoing edges
            + ([np.log(1/isc)] * isc if intron_chain_starts else [0])
            # outgoing edges
            + ([np.log(1/2) if intron_chain_transitions else 0] * (isc-1)
               if intron_chain_skips else []) + [
                np.log(1/2) if intron_chain_loop else 0
            ]
            # chain loop edges
            + ([np.log(1/2)] if intron_chain_loop else [])
        )

    return indices, values.tolist(), share


def state_transitions_coding(
    isc: int = 1,
    p_exon: int | float | None = None,
    heads: int = 1,
    share_frames: bool = True,
    spliced_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    if spliced_stop:
        add = 3
    else:
        add = 0
    indices = np.array([
        [ 3,  1, 0],
        [ 1, 10+2*add, np.log(1/2)],

        # E0 -> E1 -> E2 -> E0
        [ 0,  1, np.log(p_exon - 1)],
        [ 1,  2, np.log(p_exon - 1)],
        [ 2,  0, np.log(p_exon - 1)],

        # Ek -> EIk
        [ 0,  4, 0],
        [ 1,  5, np.log(1/2)],
        [ 2,  6, 0],
    ])
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            # Ek -> EIk
            [ 0,  8, 0],
            [ 0,  9, 0],
            [ 2,  7, 0],
        ])))
    indices = np.concatenate((indices, np.array([
        # IEk -> Ek
        [7+add, 0, 0],
        [8+add, 1, 0],
        [9+add, 2, 0],
    ])))
    if spliced_stop:
        indices = np.concatenate((indices, np.array([
            # IEk -> Ek
            [14, 0, 0],
            [15, 0, 0],
            [13, 2, 0],
        ])))

    values = indices[:, 2].astype(np.float32)
    indices = indices[:, :2].astype(np.int64)
    return indices, values.tolist()

def state_transitions(
    isc: int = 1,
    intron_chain_starts: bool = False,
    intron_chain_skips: bool = False,
    intron_chain_loop: bool = False,
    intron_chain_transitions: bool = True,
    p_IR: int | float | None = None,
    p_intron: int | float | list[float | int] | None = None,
    p_exon: int | float | None = None,
    heads: int = 1,
    share_noncoding: bool = False,
    share_frames: bool = True,
    spliced_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generates a tuple of arrays ``(indices, values, share)`` for the
    initialization of an HMM transitioner.

    Args:
        isc (int, optional): Number of intron states per frame  in the
            transitioner. These states build an intron chain and model
            the length of introns as a negative binomial distribution.
            Defaults to 1.
        intron_chain_skips (bool, optional): Whether additional outgoing
            connections within each intron chain should be created. They
            are shortcuts out of such a chain. Defaults to False.
        p_IR (float, optional): The probability of staying in the IR
            state or an integer > 1 as the expected length of an IR
            region. Defaults to `0.9999`.
        p_intron (float or list[float], optional): The probability of
            staying in the intron state or an integer > 1 as the
            expected length of an intron. For `isc > 1` this should be a
            list of probabilities (or integers), one for each intron
            state in the chain. Defaults to `0.99`.
        p_exon (float, optional): The probability of moving between the
            different exon states (or the expected length of an exon).
            Defaults to `0.975`.
        heads (int, optional): Number of heads of the HMM. This will
            only copy the created transition arrays a number of times.
            Defaults to 1.
    """
    if p_IR is None: p_IR = 0.9999
    if p_exon is None: p_exon = 0.975
    if p_intron is None: p_intron = [0.99] * isc
    elif isinstance(p_intron, (float, int)): p_intron = [p_intron] * isc

    if p_IR < 1: p_IR = 1 / (1-p_IR)
    if p_exon < 1: p_exon = 1 / (1-p_exon)
    for i in range(len(p_intron)):
        if p_intron[i] < 1: p_intron[i] = 1 / (1-p_intron[i])

    if isc == 1:
        return state_transitions_simple(
            share_frames=share_frames,
            share_noncoding=share_noncoding,
            p_exon=p_exon,
            p_intron=p_intron[0],
            p_IR=p_IR,
            heads=heads,
            spliced_stop=spliced_stop
        )
    elif share_noncoding or not share_frames:
        raise ValueError(
            "Intron state chains do not work with non-default share"
            " configurations."
        )

    if spliced_stop:
        add = 3
    else:
        add = 0
    
    indices_ir,values_ir = state_transitions_ir(p_IR, spliced_stop)
    indices_coding,values_coding = state_transitions_coding(isc,p_exon,heads,share_frames,spliced_stop)
    indices_introns,values_introns,share_introns = state_transitions_introns(isc,intron_chain_starts,
        intron_chain_skips,intron_chain_loop,intron_chain_transitions,p_intron,heads,share_frames,spliced_stop)
    
    indices_coding[indices_coding >-1] = indices_coding[indices_coding >-1]+1+(3+add)
    indices_introns[indices_introns >-1] = indices_introns[indices_introns >-1]+1
    
    indices = np.concatenate((indices_ir,indices_coding))
    
    values = np.concatenate((values_ir, values_coding))

    if isc > 1:
        indices[indices > 0] = indices[indices > 0] + (3+add)*(isc-1)

    n_edges = len(indices)
    share_introns[share_introns>-1] = share_introns[share_introns>-1] + n_edges
    
    indices = np.r_[indices, indices_introns]

    repeats = np.arange(heads).reshape(heads, 1, 1)
    repeats = np.tile(repeats, (1, len(indices), 1))
    indices = np.tile(indices, (heads, 1, 1))
    indices = np.concatenate([repeats, indices], axis=-1, dtype=np.int64)
    indices = indices.reshape(-1, 3)
   
    share = np.r_[*[
        share_introns + i*(len(indices)//heads) for i in range(heads)
    ]]

    values = np.r_[
        values,
        values_introns
    ]
    values = np.exp(values) / np.sum(np.exp(values), -1, keepdims=True)
    values = np.tile(values, heads)

    return indices.tolist(), values, share

def state_start_dist(
    isc: int = 1,
    heads: int = 1,
    spliced_stop: bool = False,
) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]]]:

    if not spliced_stop:
        intron_number = 3
        add_states = 0
    else:
        intron_number = 6
        add_states = 6

    indices = np.array([
        [h, j] for h in range(heads) for j in range(12+add_states+intron_number*isc)
    ])

    # values = np.array([
    #     np.log(100),
    #     np.log(4),
    #     np.log(10), np.log(20), np.log(10),
    #     0,
    # ], dtype=np.float32)
    values = np.random.normal(loc=0, scale=0.01, size=6)
    values = np.exp(values) / np.sum(np.exp(values), -1, keepdims=True)
    values = np.tile(values, heads)

    share = np.array([
        [1, 1+intron_number*isc], [4+intron_number*isc,12+add_states+intron_number*isc]
    ])
    share = np.concatenate(
        [share + i*(12+add_states+intron_number*isc) for i in range(heads)],
        axis=0,
    )
 
    return indices.tolist(), values, share.tolist()


def state_names(isc: int = 1, spliced_stop: bool = False,) -> list[str]:
    return (
        ["IR"]
        + [f"I{k}{j}" for j in range(isc) for k in (range(6) if spliced_stop else range(3))]
        + ["E0", "E1", "E2"]
        + ["START", "EI0", "EI1", "EI2"] 
        + ([ "EI3", "EI4", "EI5"] if spliced_stop else [])
        + ["IE0", "IE1", "IE2"]
        + ([ "IE3", "IE4", "IE5"] if spliced_stop else []) 
        + [ "STOP"]
    )
