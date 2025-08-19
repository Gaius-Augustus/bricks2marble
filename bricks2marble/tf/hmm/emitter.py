from typing import Self

import tensorflow as tf

from ..config import ModelConfig, with_config
from .tools import get_nuc_emission_distribution, make_kmer


class EmitterConfig(ModelConfig):

    start_codons: list[tuple[str, float]]
    stop_codons: list[tuple[str, float]]
    intron_begin_pattern: list[tuple[str, float]]
    intron_end_pattern: list[tuple[str, float]]
    intron_state_chain: int = 1

    heads: int = 1
    use_reverse_strand: bool = False
    sigmoid_activation: bool = False

    share_noncoding_params: bool = False

    @property
    def n_states(self) -> int:
        return 12 + 3*self.intron_state_chain


@with_config(EmitterConfig)
class Emitter(tf.keras.layers.Layer):
    """Defines the emission probabilities for a gene prediction HMM
    with embeddings or class predictions as inputs. Extends the simple
    HMM with start- and stop-states that enforce biological structure.
    """

    config: EmitterConfig

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.emission_kernel = self.add_weight(
            shape=[
                self.config.heads,
                self.config.n_states if not self.config.share_noncoding_params
                    else self.config.n_states-3*self.config.intron_state_chain,
                input_shape[-1],
            ],
            initializer=tf.initializers.GlorotNormal(),
            name="emission_kernel",
        )

        self.codon_probs = tf.Variable(
            get_nuc_emission_distribution(
                start_codons=self.config.start_codons,
                stop_codons=self.config.stop_codons,
                intron_begin_pattern=self.config.intron_begin_pattern,
                intron_end_pattern=self.config.intron_end_pattern,
            ),
            trainable=False,
        )

    def recurrent_init(self) -> None:
        self.B = self.make_B()

    def make_B(self):
        if self.config.share_noncoding_params:
            B = tf.concat(
                [self.emission_kernel[:, :1, :]]
                + [
                    self.emission_kernel[:, :1, :],
                    self.emission_kernel[:, :1, :],
                    self.emission_kernel[:, :1, :],
                ] * self.config.intron_state_chain
                + [self.emission_kernel[:, 1:, :]]
            , axis=1)
            if self.config.sigmoid_activation:
                return tf.nn.sigmoid(B)
            return tf.nn.softmax(B)
        if self.config.sigmoid_activation:
            return tf.nn.sigmoid(self.emission_kernel)
        return tf.nn.softmax(self.emission_kernel)

    def call(
        self,
        inputs: tf.Tensor,
        end_hints: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        x, nuc = inputs[..., :-5], inputs[..., -5:]  #  type: ignore

        left_3mers = make_kmer(
            nuc,  # type: ignore
            k=3,
            pivot_left=True,
            collapse_pivot=True,
        )
        right_3mers = make_kmer(
            nuc,  # type: ignore
            k=3,
            pivot_left=False,
            collapse_pivot=True,
        )
        input_3mers = tf.stack([left_3mers, right_3mers], axis=-2)

        codon_emit = tf.einsum(
            "...rs,rqs->...rq",
            input_3mers,
            self.codon_probs,
        )
        codon_emit = tf.reduce_prod(codon_emit, axis=-2)
        codon_emit = tf.concat([
            tf.ones(tf.concat((
                tf.shape(codon_emit)[:-1],  # type: ignore
                [3 + 3*self.config.intron_state_chain],
            ), axis=0)) / 4096.,
            codon_emit,
        ], axis=-1)
        if training:
            codon_emit += 1e-7  # type: ignore

        emit = tf.einsum("...s,hqs->h...q", x[0], self.B)
        if self.config.sigmoid_activation:
            emit /= self.emission_kernel.shape[-1]

        return emit * codon_emit

    def get_prior_log_density(self) -> list[list[float]]:
        return [[0.]]

    def get_aux_loss(self) -> float:
        return 0.

    def duplicate(
        self,
        model_indices=None,
        share_kernels: bool = False,
    ) -> Self:
        emitter_copy = Emitter.from_config(self.get_config())
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
        return emitter_copy
