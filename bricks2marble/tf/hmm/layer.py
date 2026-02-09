from typing import Literal

import tensorflow as tf
from hidten import HMMMode
from hidten.config import ModelConfig, with_config
from hidten.tf import TFHMM, TFBernoulliEmitter, TFCategoricalEmitter

from ..loss import (IntronParameterRegularizer, IRIntronRatioRegularizer,
                    RepeatsNonCodingRegularizer,
                    UncertainPredictionRegularizer)
from .tools import (emission_parameters, emission_parameters_eye,
                    get_nuc_emission_distribution,
                    get_repeats_at_borders_multiplier,
                    get_repeats_emission_distribution, left_right_3mers,
                    state_names, state_start_dist, state_transitions)


class TransitionScorerConfig(ModelConfig):

    input_size: int | None = None
    activation: str = "swish"
    latent_factor: float = 2.0
    out_activation: str | None = None

    norm: Literal["layer", "batch"] | None = None
    regularization: float | None = None


class TransitionScorer(tf.keras.Layer):
    """Layer that calculates the transition deltas which are added on
    top of the kernel of the transition matrix:
    ```
        A = softmax(kernel + delta(x))
    ```

    Currently, this layer is only configured for learnable transition
    matrices with one intron state per frame.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = TransitionScorerConfig(**kwargs)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        if self.config.input_size is None:
            raise RuntimeError(
                "Undetermined input size to TransitionScorer. Has to be "
                "manually given by super-layers."
            )
        if self.config.norm == "layer":
            self.norm = tf.keras.layers.LayerNormalization()
        elif self.config.norm == "batch":
            self.norm = tf.keras.layers.BatchNormalization()
        if self.config.norm is not None:
            self.norm.build(input_shape[:-1] + (self.config.input_size, ))
        d_latent = int(self.config.latent_factor * self.config.input_size)

        self.f1 = tf.keras.layers.Dense(
            units=d_latent,
            use_bias=True,
            activation=self.config.activation,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.f1.build(input_shape[:-1] + (self.config.input_size, ))
        self.f2 = tf.keras.layers.Dense(
            units=7,
            use_bias=False,
            activation=self.config.out_activation,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.f2.build(input_shape[:-1] + (d_latent, ))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, T, _ = tf.unstack(tf.shape(x))
        if self.config.norm is not None:
            x = self.norm(x)
        scores = self.f2(self.f1(x))
        if self.config.regularization is not None:
            self.add_loss(
                self.config.regularization * tf.reduce_mean(
                    tf.sqrt(tf.reduce_sum(scores**2, axis=-1))
                )
            )
        return tf.concat((
            scores[:, :, :4],
            tf.zeros((B, T, 10), dtype=scores.dtype),
            scores[:, :, 4:],
            tf.zeros((B, T, 6), dtype=scores.dtype),
        ), axis=-1)


class AnnotationHMMConfig(ModelConfig):

    start_codons: list[tuple[str, float]] = [("ATG", 1.)]
    stop_codons: list[tuple[str, float]] = [
        ("TAG", .34), ("TAA", .33), ("TGA", .33),
    ]
    intron_begin_pattern: list[tuple[str, float]] = [
        ("NGT", 0.99), ("NGC", 0.01),
    ]
    intron_end_pattern: list[tuple[str, float]] = [("AGN", 1.)]

    heads: int = 1
    dropout_heads: float = 0
    compute_heads_sequentially: bool = False
    use_reverse_strand: bool = False

    emitter_share_noncoding: bool = False
    """Shares noncoding state emission parameters."""
    emitter_share_frames: bool = True
    """Shares state emissions that correspond to the same state in
    different reading frames."""
    emitter_sigmoid_activation: bool = False
    """Uses a sigmoid activated emission matrix instead of softmax
    activated one."""
    emitter_eye: float | None = None
    """Replaces the stream emitter by an identity matrix that has an
    epsilon removed from the main diagonal and distributed to all other
    columns."""
    train_emitter: bool = True
    """Toggles training of the emitter."""
    repeats_emitter: float | None = None

    intron_state_chain: int = 1
    intron_chain_skips: bool = False
    intron_chain_loop: bool = False
    initial_exon_len: int | float | None = None
    initial_intron_len: int | float | list[float | int] | None = None
    initial_ir_len: int | float | None = None
    train_transitioner: bool = True
    transitioner_share_noncoding: bool = False
    transitioner_share_frames: bool = True
    transition_scorer: TransitionScorerConfig | None = None
    uniform_N: bool = False
    nudge_IR: float = 0.0
    nudge_repeats_noncoding: float = 0.0
    intron_regularization: float = 0.0
    ir_intron_ratio_regularization: float = 0.0

    repeats_at_borders: float | None = None

    @property
    def repeats_in_nuc(self) -> bool:
        return (
            self.nudge_repeats_noncoding > 0
            or self.repeats_at_borders is not None
            or self.repeats_emitter is not None
        )

    @property
    def n_states(self) -> int:
        return 12 + 3*self.intron_state_chain

    model_config = {"frozen": True, "extra": "forbid"}


@with_config(AnnotationHMMConfig)
class AnnotationHMM(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = AnnotationHMMConfig(**kwargs)

        heads = (1 if self.config.compute_heads_sequentially
                 else self.config.heads)

        transitions, values_transitions, share_transitions = state_transitions(
            isc=self.config.intron_state_chain,
            intron_chain_skips=self.config.intron_chain_skips,
            intron_chain_loop=self.config.intron_chain_loop,
            p_IR=self.config.initial_ir_len,
            p_intron=self.config.initial_intron_len,
            p_exon=self.config.initial_exon_len,
            heads=heads,
            share_frames=self.config.transitioner_share_frames,
            share_noncoding=self.config.transitioner_share_noncoding,
        )
        starts, values_starts, share_starts = state_start_dist(
            isc=self.config.intron_state_chain,
            heads=heads,
        )
        emissions_left, emissions_right = get_nuc_emission_distribution(
            start_codons=self.config.start_codons,
            stop_codons=self.config.stop_codons,
            intron_begin_pattern=self.config.intron_begin_pattern,
            intron_end_pattern=self.config.intron_end_pattern,
            intron_state_chain=self.config.intron_state_chain,
            heads=heads,
        )
        if self.config.repeats_emitter is not None:
            emissions_repeats = get_repeats_emission_distribution(
                self.config.repeats_emitter,
                intron_state_chain=self.config.intron_state_chain,
                heads=heads,
            )

        nhmms = 1
        if self.config.compute_heads_sequentially:
            nhmms = self.config.heads
            self.hmm = []

        for _ in range(nhmms):
            hmm = TFHMM(
                states=self.config.n_states,
                heads=heads,
            )

            hmm.transitioner.allow = transitions
            hmm.transitioner.share = share_transitions
            hmm.transitioner.initializer = values_transitions

            hmm.transitioner.allow_start = starts
            hmm.transitioner.share_start = share_starts
            hmm.transitioner.initializer_start = values_starts

            hmm.transitioner.trainable = self.config.train_transitioner

            if (
                self.config.emitter_sigmoid_activation
                and self.config.emitter_eye is None
            ):
                stream_emitter = TFBernoulliEmitter()
            else:
                stream_emitter = TFCategoricalEmitter()

            nuc_emitter_left = TFCategoricalEmitter()
            nuc_emitter_right = TFCategoricalEmitter()

            nuc_emitter_left.initializer = emissions_left.flatten()
            nuc_emitter_left.trainable = False
            nuc_emitter_right.initializer = emissions_right.flatten()
            nuc_emitter_right.trainable = False

            hmm.add_emitter(stream_emitter)
            hmm.add_emitter(nuc_emitter_left)
            hmm.add_emitter(nuc_emitter_right)

            if self.config.repeats_emitter is not None:
                repeats_emitter = TFBernoulliEmitter()
                repeats_emitter.initializer = emissions_repeats.flatten()
                repeats_emitter.trainable = False
                hmm.add_emitter(repeats_emitter)

            if self.config.compute_heads_sequentially:
                self.hmm.append(hmm)
            else:
                self.hmm = hmm

        if self.config.transition_scorer is not None:
            if (
                self.config.transitioner_share_frames
                or self.config.transitioner_share_noncoding
                or not self.config.train_transitioner
            ):
                raise ValueError(
                    "Input dependent transitions are currently only"
                    " implemented for independently trained transitions."
                )
            self.transition_scorer = TransitionScorer(
                **self.config.transition_scorer.model_dump()
            )

        if self.config.repeats_at_borders is not None:
            leftrb, rightrb = get_repeats_at_borders_multiplier(
                self.config.repeats_at_borders,
                self.config.start_codons,
                self.config.intron_begin_pattern,
                self.config.stop_codons,
                self.config.intron_end_pattern,
            )
            self.left_repeats_borders = tf.Variable(leftrb, trainable=False)
            self.right_repeats_borders = tf.Variable(rightrb, trainable=False)

        if self.config.nudge_IR > 0:
            self.regularizer = UncertainPredictionRegularizer(
                weight=self.config.nudge_IR,
                class_index=0,
            )
        if self.config.nudge_repeats_noncoding > 0:
            self.repeats_regularizer = RepeatsNonCodingRegularizer(
                weight=self.config.nudge_repeats_noncoding,
                coding_start_index=1+3*self.config.intron_state_chain,
            )
        if self.config.ir_intron_ratio_regularization > 0:
            self.ir_intron_regularizer = IRIntronRatioRegularizer(
                weight=self.config.ir_intron_ratio_regularization,
                intron_state_chain=self.config.intron_state_chain,
            )
        if self.config.dropout_heads > 0:
            self.dropout = tf.keras.layers.Dropout(self.config.dropout_heads)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        D: int = input_shape[-1]  # type: ignore
        S = self.config.n_states
        H = 1 if self.config.compute_heads_sequentially else self.config.heads
        isc = self.config.intron_state_chain

        if self.config.emitter_eye is not None:
            init, allow, share = emission_parameters_eye(
                D=D, S=S, H=H, epsilon=self.config.emitter_eye,
            )
        else:
            init, allow, share = emission_parameters(
                D=D, S=S, H=H, isc=isc,
                share_noncoding=self.config.emitter_share_noncoding,
                share_frames=self.config.emitter_share_frames,
            )
        for hmm in (
            self.hmm if self.config.compute_heads_sequentially else [self.hmm]
        ):
            hmm.emitter[0].allow = allow
            if share is not None:
                hmm.emitter[0].share = share
            hmm.emitter[0].initializer = init
            hmm.emitter[0].trainable = self.config.train_emitter
            if self.config.repeats_emitter is None:
                hmm.build((
                    input_shape,
                    input_shape[:-1] + (65, ),
                    input_shape[:-1] + (65, ),
                ))
            else:
                hmm.build((
                    input_shape,
                    input_shape[:-1] + (65, ),
                    input_shape[:-1] + (65, ),
                    input_shape[:-1] + (2, ),
                ))

        if self.config.intron_regularization > 0:
            if self.config.compute_heads_sequentially:
                matrix = tf.concat([
                    hmm.transitioner.matrix() for hmm in self.hmm
                ], axis=0)
            else:
                matrix = self.hmm.transitioner.matrix()
            isc = self.config.intron_state_chain
            intron_value = tf.concat([tf.reduce_mean(
                tf.linalg.diag_part(matrix[i])[1:1+3*isc],
                keepdims=True,
            ) for i in range(self.config.heads)], axis=0)
            self.intron_regularizer = IntronParameterRegularizer(
                weight=self.config.intron_regularization,
                intron_state_chain=self.config.intron_state_chain,
                start_value=intron_value,
            )
        if self.config.transition_scorer is not None:
            self.transition_scorer.build(input_shape)

    def state_names(self) -> list[str]:
        return state_names(self.config.intron_state_chain)

    def preprocess(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
        transition_scores: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
        if self.config.repeats_in_nuc:
            five = 4 if self.config.uniform_N else 5
            r = nuc[..., five:five+1]
            nuc = nuc[..., :five]
        if self.config.use_reverse_strand:
            nuc_reverse = tf.gather(
                nuc,
                [3, 2, 1, 0] if self.config.uniform_N else [3, 2, 1, 0, 4],
                axis=-1,
            )
            nuc_reverse = tf.reverse(nuc_reverse, [-2])
            nuc = tf.concat((nuc, nuc_reverse), axis=0)  # type: ignore
            x = tf.concat((x, tf.reverse(x, [-2])), axis=0)  # type: ignore
            if transition_scores is not None:
                transition_scores = tf.concat((
                    transition_scores,
                    tf.reverse(transition_scores, [-2]),
                ), axis=0)  # type: ignore
        if self.config.repeats_in_nuc:
            if self.config.repeats_emitter is not None:
                r = tf.concat((r, tf.reverse(r, [1])), 0)
                r = tf.stack([1.0 - r, r], axis=-1)
            return x, nuc, r, transition_scores
        return x, nuc, None, transition_scores

    def postprocess(
        self,
        x: tf.Tensor,
        mode: HMMMode = HMMMode.POSTERIOR,
        training: bool = False,
    ) -> tf.Tensor:
        """If `use_reverse_strand` is active, the resulting tensor ``x``
        will be of shape ``(B, T, 2*H)`` or ``(B, T, 2*H, D)``, where
        ``x[:, :, :H, ...]`` is the output of the forward strand and
        ``x[:, :, H:, ...]`` is the output of the backward strand.
        However, both series are pointing in the same direction.
        """
        if self.config.use_reverse_strand:
            # 2*B, T, H(, D)
            x = tf.reshape(x, tf.concat((
                (2, ),
                (tf.shape(x)[0]//2, ),  # type: ignore
                tf.shape(x)[1:]  # type: ignore
            ), 0))
            # 2, B, T, H(, D)
            x = tf.concat(
                (x[0:1], tf.reverse(x[1:2], [2])  # type: ignore
            ), 0)
            x = tf.keras.ops.moveaxis(x, 0, 2)  # type: ignore
            # B, T, 2, H(, D)
            x = tf.reshape(x, tf.concat((
                tf.shape(x)[:2],  # type: ignore
                (tf.shape(x)[2]*tf.shape(x)[3], ),  # type: ignore
                tf.shape(x)[4:],  # type: ignore
            ), 0))
            # B, T, 2*H(, D)
        if mode == HMMMode.POSTERIOR and self.config.dropout_heads > 0:
            B, H = tf.shape(x)[0], tf.shape(x)[2]
            mask = tf.ones((B, H), dtype=x.dtype)
            mask = self.dropout(mask, training=training)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1)
            mask = tf.broadcast_to(mask, tf.shape(x))
            x = mask * x
        return x

    def call_HMM(
        self,
        x: tf.Tensor,
        nuc_left: tf.Tensor,
        nuc_right: tf.Tensor,
        repeats: tf.Tensor | None = None,
        mode: HMMMode | Literal["A"] = HMMMode.POSTERIOR,
        parallel: int = 1,
        transition_scores: tf.Tensor | None = None,
    ) -> tf.Tensor:
        if self.config.compute_heads_sequentially:
            if self.config.use_reverse_strand:
                B = tf.shape(x)[0] // 2
                return tf.concat([
                    tf.concat([
                        hmm(
                            x[:B],
                            nuc_left[:B],
                            nuc_right[:B],
                            mode=mode,
                            parallel=parallel,
                        )
                        for hmm in self.hmm
                    ], axis=2),
                    tf.concat([
                        hmm(
                            x[B:],
                            nuc_left[B:],
                            nuc_right[B:],
                            mode=mode,
                            parallel=parallel,
                        )
                        for hmm in self.hmm
                    ], axis=2),
                ], axis=0)  # type: ignore
            return tf.concat([
                hmm(x, nuc_left, nuc_right, mode=mode, parallel=parallel)
                for hmm in self.hmm
            ], axis=2)  # type: ignore

        if self.config.transition_scorer is not None:
            scores = self.transition_scorer(transition_scores)
        else:
            scores = None
        if mode == "A":
            return self.hmm.transitioner.matrix(transition_delta=scores)
        if self.config.repeats_emitter is not None:
            return self.hmm(
                x, nuc_left, nuc_right, repeats,
                transition_delta=scores,
                mode=mode,
                parallel=parallel,
            )  # type: ignore
        return self.hmm(
            x, nuc_left, nuc_right,
            transition_delta=scores,
            mode=mode,
            parallel=parallel,
        )  # type: ignore

    def call(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
        mode: HMMMode = HMMMode.POSTERIOR,
        parallel: int = 1,
        training: bool = False,
        transition_scores: tf.Tensor | None = None,
    ) -> tf.Tensor:
        if (
            transition_scores is None
            and self.config.transition_scorer is not None
        ):
            transition_scores = x
        x, nuc, r, transition_scores = self.preprocess(
            x, nuc, transition_scores,
        )
        nuc_left, nuc_right = left_right_3mers(
            nuc,
            uniform_N=self.config.uniform_N,
            repeats=r if self.config.repeats_at_borders is not None else None,
            left_factor=(
                self.left_repeats_borders
                if self.config.repeats_at_borders is not None else None
            ),
            right_factor=(
                self.right_repeats_borders
                if self.config.repeats_at_borders is not None else None
            ),
        )
        x = self.call_HMM(
            x,
            nuc_left, nuc_right,
            repeats=r if self.config.repeats_emitter is not None else None,
            mode=mode,
            parallel=parallel,
            transition_scores=transition_scores,
        )
        x = self.postprocess(x, mode=mode, training=training)
        if self.config.nudge_IR > 0 and training: self.regularizer(x)
        if self.config.nudge_repeats_noncoding > 0 and training:
            self.repeats_regularizer(x, r)
        if self.config.intron_regularization > 0 and training:
            if self.config.compute_heads_sequentially:
                matrix = tf.concat([
                    hmm.transitioner.matrix() for hmm in self.hmm
                ], axis=0)
            else:
                matrix = self.hmm.transitioner.matrix()
            self.intron_regularizer(matrix)
        if self.config.ir_intron_ratio_regularization > 0 and training:
            if self.config.compute_heads_sequentially:
                matrix = tf.concat([
                    hmm.transitioner.matrix() for hmm in self.hmm
                ], axis=0)
            else:
                matrix = self.hmm.transitioner.matrix()
            self.ir_intron_regularizer(matrix)
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape[:-1] + (
            (2 if self.config.use_reverse_strand else 1) * self.config.heads,
            self.config.n_states,
        )
