from typing import Any, Self

import numpy as np
import tensorflow as tf

from ..config import ModelConfig, with_config
from ..util import shared_sparse_tensor
from .tools import state_start_dist, state_transitions

STATES = ["IR", "I0", "I1", "I2", "E0", "E1", "E2",
          "START", "EI0", "EI1", "EI2", "IE0", "IE1", "IE2", "STOP"]


class TransitionerConfig(ModelConfig):

    heads: int = 1
    intron_state_chain: int = 1
    initial_exon_len: int = 100
    initial_intron_len: int = 10000
    initial_ir_len: int = 10000
    starting_distribution_trainable: bool = True
    transitions_trainable: bool = True
    init_component_sd: float = 0.05

    @property
    def n_states(self) -> int:
        return 12 + 3*self.intron_state_chain


@with_config(TransitionerConfig)
class Transitioner(tf.keras.layers.Layer):
    """Defines which transitions between HMM states are allowed and how
    they are initialized.
    """

    config: TransitionerConfig

    def post_config_init(self) -> None:
        super().__init__()
        self._indices, self._values, self._shared_values = state_transitions(
            isc=self.config.intron_state_chain,
            T_exon=self.config.initial_exon_len,
            T_intron=self.config.initial_intron_len,
            T_ir=self.config.initial_ir_len,
            heads=self.config.heads,
        )
        self._indices_st, self._values_st, self._shared_values_st = (
            state_start_dist(
                isc=self.config.intron_state_chain,
                heads=self.config.heads,
            )
        )
        self.reverse = False

    def cell_init(self, cell) -> None:
        pass

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        if self.built:
            return

        self.transition_kernel = self.add_weight(
            shape=[self.config.heads, len(self._values) // self.config.heads],
            initializer=tf.keras.initializers.Constant(
                self._values.reshape(self.config.heads, -1),  # type: ignore
            ),
            trainable=self.config.transitions_trainable,
            name="transition_kernel",
        )
        self.starting_distribution_kernel = self.add_weight(
            shape=[
                1,
                self.config.heads,
                len(self._values_st) // self.config.heads,
            ],
            initializer=tf.keras.initializers.Constant(np.reshape(
                self._values_st,
                (1, self.config.heads, -1),
            )),  # type: ignore
            name="starting_distribution_kernel",
            trainable=self.config.starting_distribution_trainable,
        )
        self.built = True

    def recurrent_init(self) -> None:
        self.A = self.make_A()
        self.A_transposed = tf.transpose(self.A, (0, 2, 1))

    def make_A_sparse(self) -> tf.SparseTensor:
        tensor = shared_sparse_tensor(
            indices=self._indices,
            values=tf.reshape(self.transition_kernel, [-1]),
            shape=tf.constant([
                self.config.heads,
                self.config.n_states,
                self.config.n_states,
            ], dtype=tf.int64),
            share=self._shared_values,
        )
        return tf.sparse.softmax(tensor)

    def make_A(self) -> tf.Tensor:
        return tf.sparse.to_dense(self.make_A_sparse())

    def make_log_A(self) -> tf.Tensor:
        A_sparse = self.make_A_sparse()
        log_A_sparse = tf.sparse.map_values(tf.math.log, A_sparse)
        log_A = tf.sparse.to_dense(log_A_sparse, default_value=-1e3)
        return log_A

    def make_initial_distribution(self) -> tf.Tensor:
        tensor = shared_sparse_tensor(
            indices=self._indices_st,
            values=tf.reshape(self.starting_distribution_kernel, [-1]),
            shape=tf.constant([
                1,
                self.config.heads,
                self.config.n_states,
            ], dtype=tf.int64),
            share=self._shared_values_st,
        )
        return tf.sparse.to_dense(tf.sparse.softmax(tensor))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.reverse:
            return tf.matmul(inputs, self.A_transposed)
        else:
            return tf.matmul(inputs, self.A)

    def get_prior_log_densities(self) -> dict[str | int, Any]:
        return {"none" : 0.}

    def duplicate(
        self,
        model_indices=None,
        share_kernels: bool = False,
    ) -> Self:
        transitioner_copy = Transitioner.from_config(self.get_config())
        if share_kernels:
            transitioner_copy.transition_kernel = self.transition_kernel
            transitioner_copy.starting_distribution_kernel = (
                self.starting_distribution_kernel
            )
            transitioner_copy.built = True
        return transitioner_copy
