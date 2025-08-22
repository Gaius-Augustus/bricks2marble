import tensorflow as tf
from hidten import HMMMode
from hidten.config import ModelConfig, with_config
from hidten.tf import TFHMM, TFBernoulliEmitter, TFCategoricalEmitter

from ..loss import UncertainPredictionRegularizer
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
    use_reverse_strand: bool = False
    parallel_factor: int = 1

    emitter_sigmoid_activation: bool = False

    initial_exon_len: int | None = None
    initial_intron_len: int | None = None
    initial_ir_len: int | None = None
    intron_state_chain: int = 1
    train_transitions: bool = True
    train_start_dist: bool = True
    share_noncoding_params: bool = False
    nudge_IR: float = 0.0

    @property
    def n_states(self) -> int:
        return 12 + 3*self.intron_state_chain

    model_config = {"frozen": True, "extra": "forbid"}


@with_config(AnnotationHMMConfig)
class AnnotationHMM(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = AnnotationHMMConfig(**kwargs)

        self.hmm = TFHMM(
            states=self.config.n_states,
            heads=self.config.heads,
        )

        transitions, values, share = state_transitions(
            isc=self.config.intron_state_chain,
            T_exon=self.config.initial_exon_len,
            T_intron=self.config.initial_intron_len,
            T_ir=self.config.initial_ir_len,
            heads=self.config.heads,
        )

        self.hmm.transitioner.allow = transitions
        self.hmm.transitioner.share = share
        self.hmm.transitioner.initializer = values

        starts, values, share = state_start_dist(
            isc=self.config.intron_state_chain,
            heads=self.config.heads,
        )

        self.hmm.transitioner.allow_start = starts
        self.hmm.transitioner.share_start = share
        self.hmm.transitioner.initializer_start = values

        if self.config.emitter_sigmoid_activation:
            stream_emitter = TFBernoulliEmitter()
        else:
            stream_emitter = TFCategoricalEmitter()

        nuc_emitter_left = TFCategoricalEmitter()
        nuc_emitter_right = TFCategoricalEmitter()

        emissions_left, emissions_right = get_nuc_emission_distribution(
            start_codons=self.config.start_codons,
            stop_codons=self.config.stop_codons,
            intron_begin_pattern=self.config.intron_begin_pattern,
            intron_end_pattern=self.config.intron_end_pattern,
            intron_state_chain=self.config.intron_state_chain,
            heads=self.config.heads,
        )

        nuc_emitter_left.initializer = emissions_left.flatten()
        nuc_emitter_left.trainable = False
        nuc_emitter_right.initializer = emissions_right.flatten()
        nuc_emitter_right.trainable = False

        self.hmm.add_emitter(stream_emitter)
        self.hmm.add_emitter(nuc_emitter_left)
        self.hmm.add_emitter(nuc_emitter_right)

        if self.config.nudge_IR > 0:
            self.regularizer = UncertainPredictionRegularizer(
                weight=self.config.nudge_IR,
                class_index=0,
            )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        D: int = input_shape[-1]  # type: ignore
        S = self.config.n_states
        H = self.config.heads
        isc = self.config.intron_state_chain
        self.hmm.emitter[0].allow = [
            (h, i, k)
            for h, states in enumerate([S]*H)
            for k in range(D)
            for i in range(states)
        ]
        self.hmm.emitter[0].share = ([
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
        self.hmm.emitter[0].initializer = tf.initializers.GlorotNormal()
        self.hmm.build((
            input_shape,
            input_shape[:-1] + (65, ),
            input_shape[:-1] + (65, ),
        ))

    def state_names(self) -> list[str]:
        return state_names(self.config.intron_state_chain)

    def preprocess(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if self.config.use_reverse_strand:
            nuc_reverse = tf.gather(nuc, [3, 2, 1, 0, 4], axis=-1)
            nuc_reverse = tf.reverse(nuc_reverse, [-2])
            nuc = tf.concat((nuc, nuc_reverse), axis=0)  # type: ignore
            x = tf.concat((x, tf.reverse(x, [-2])), axis=0)  # type: ignore
        return x, nuc

    def postprocess(self, x: tf.Tensor) -> tf.Tensor:
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
        return x

    def call_HMM(
        self,
        x: tf.Tensor,
        nuc_left: tf.Tensor,
        nuc_right: tf.Tensor,
        mode: HMMMode = HMMMode.POSTERIOR,
        parallel: int = 1,
    ) -> tf.Tensor:
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
    ) -> tf.Tensor:
        x, nuc = self.preprocess(x, nuc)
        nuc_left, nuc_right = left_right_3mers(nuc)  # type: ignore
        x = self.call_HMM(x, nuc_left, nuc_right, mode=mode, parallel=parallel)
        x = self.postprocess(x)
        if self.config.nudge_IR > 0: self.regularizer(x)
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape[:-1] + (
            (2 if self.config.use_reverse_strand else 1) * self.config.heads,
            self.config.n_states,
        )
