import tensorflow as tf
from hidten import HMMMode
from hidten.config import ModelConfig, with_config
from hidten.tf import TFHMM, TFBernoulliEmitter, TFCategoricalEmitter

from ..loss import (IntronParameterRegularizer, RepeatsNonCodingRegularizer,
                    UncertainPredictionRegularizer)
from .tools import (get_nuc_emission_distribution, left_right_3mers,
                    state_names, state_start_dist, state_transitions)


class AnnotationHMMConfig(ModelConfig):

    start_codons: list[tuple[str, float]] = [("ATG", 1.)]
    stop_codons: list[tuple[str, float]] = [
        ("TAG", .34), ("TAA", .33), ("TGA", .33),
    ]
    intron_begin_pattern: list[tuple[str, float]] = [("NGT", 1.)]
    intron_end_pattern: list[tuple[str, float]] = [("AGN", 1.)]

    heads: int = 1
    dropout_heads: float = 0
    compute_heads_sequentially: bool = False
    use_reverse_strand: bool = False
    parallel_factor: int = 1

    emitter_sigmoid_activation: bool = False

    intron_state_chain: int = 1
    intron_chain_skips: bool = False
    intron_chain_loop: bool = False
    initial_exon_len: int | float | None = None
    initial_intron_len: int | float | list[float | int] | None = None
    initial_ir_len: int | float | None = None
    train_transitioner: bool = True
    share_noncoding_params: bool = False
    uniform_N: bool = False
    nudge_IR: float = 0.0
    nudge_repeats_noncoding: float = 0.0
    intron_regularization: float = 0.0

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

            if self.config.emitter_sigmoid_activation:
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

            if self.config.compute_heads_sequentially:
                self.hmm.append(hmm)
            else:
                self.hmm = hmm

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
        if self.config.dropout_heads > 0:
            self.dropout = tf.keras.layers.Dropout(self.config.dropout_heads)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        D: int = input_shape[-1]  # type: ignore
        S = self.config.n_states
        H = 1 if self.config.compute_heads_sequentially else self.config.heads
        isc = self.config.intron_state_chain
        for hmm in (
            self.hmm if self.config.compute_heads_sequentially else [self.hmm]
        ):
            hmm.emitter[0].allow = [
                (h, i, k)
                for h, states in enumerate([S]*H)
                for k in range(D)
                for i in range(states)
            ]
            hmm.emitter[0].share = ([
                (h*D*S+i*S+1+j*3, h*D*S+i*S+4+j*3)
                for h in range(H)
                for i in range(D)
                for j in range(isc)
            ] if not self.config.share_noncoding_params else [
                (h*D*S+i*S, h*D*S+i*S+1+isc*3)
                for h in range(H)
                for i in range(D)
            ]) + [
                (h*D*S+i*S+5+3*isc, h*D*S+i*S+8+3*isc)
                for h in range(H)
                for i in range(D)
            ] + [
                (h*D*S+i*S+8+3*isc, h*D*S+i*S+11+3*isc)
                for h in range(H)
                for i in range(D)
            ]
            hmm.emitter[0].initializer = tf.initializers.GlorotNormal()
            hmm.build((
                input_shape,
                input_shape[:-1] + (65, ),
                input_shape[:-1] + (65, ),
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

    def state_names(self) -> list[str]:
        return state_names(self.config.intron_state_chain)

    def preprocess(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None]:
        if self.config.nudge_repeats_noncoding > 0:
            r = nuc[..., 5:6]
            nuc = nuc[..., :5]
        if self.config.use_reverse_strand:
            nuc_reverse = tf.gather(
                nuc,
                [3, 2, 1, 0] if self.config.uniform_N else [3, 2, 1, 0, 4],
                axis=-1,
            )
            nuc_reverse = tf.reverse(nuc_reverse, [-2])
            nuc = tf.concat((nuc, nuc_reverse), axis=0)  # type: ignore
            x = tf.concat((x, tf.reverse(x, [-2])), axis=0)  # type: ignore
        if self.config.nudge_repeats_noncoding > 0:
            return x, nuc, r
        return x, nuc, None

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
        mode: HMMMode = HMMMode.POSTERIOR,
        parallel: int = 1,
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
        return self.hmm(
            x, nuc_left, nuc_right,
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
    ) -> tf.Tensor:
        x, nuc, r = self.preprocess(x, nuc)
        nuc_left, nuc_right = left_right_3mers(
            nuc,
            uniform_N=self.config.uniform_N,
        )
        x = self.call_HMM(x, nuc_left, nuc_right, mode=mode, parallel=parallel)
        x = self.postprocess(x, mode=mode, training=training)
        if self.config.nudge_IR > 0 and training: self.regularizer(x)
        if self.config.nudge_repeats_noncoding > 0 and training:
            self.repeats_regularizer(x, r)
        if self.config.intron_regularization > 0:
            if self.config.compute_heads_sequentially:
                matrix = tf.concat([
                    hmm.transitioner.matrix() for hmm in self.hmm
                ], axis=0)
            else:
                matrix = self.hmm.transitioner.matrix()
            self.intron_regularizer(matrix)
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape[:-1] + (
            (2 if self.config.use_reverse_strand else 1) * self.config.heads,
            self.config.n_states,
        )
