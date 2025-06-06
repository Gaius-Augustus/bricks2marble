from typing import Any, Self

import numpy as np
import tensorflow as tf

from ..config import ModelConfig, with_config
from .tools import (is_exon_1_out_transition, is_exon_transition,
                    is_intergenic_loop, is_intergenic_out_transition,
                    is_intron_loop)

STATES = ["IR", "I0", "I1", "I2", "E0", "E1", "E2",
          "START", "EI0", "EI1", "EI2", "IE0", "IE1", "IE2", "STOP"]


class TransitionerConfig(ModelConfig):

    heads: int = 1
    initial_exon_len: int = 100
    initial_intron_len: int = 10000
    initial_ir_len: int = 10000
    starting_distribution_trainable: bool = True
    transitions_trainable: bool = True
    init_component_sd: float = 0.05


@with_config(TransitionerConfig)
class Transitioner(tf.keras.layers.Layer):
    """Defines which transitions between HMM states are allowed and how
    they are initialized.
    """

    config: TransitionerConfig

    def post_config_init(self) -> None:
        super().__init__()
        self.num_states = 15
        self.indices = self.make_transition_indices()
        self.num_transitions = len(self.indices) // self.config.heads
        self.init = "zeros"
        self.reverse = False

    def cell_init(self, cell) -> None:
        pass

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        if self.built:
            return
        self.transition_kernel = self.add_weight(
            shape=[self.config.heads, self.num_transitions],
            initializer=tf.keras.initializers.Constant(
                self.make_transition_init(),  # type: ignore
            ),
            trainable=self.config.transitions_trainable,
            name="transition_kernel",
        )
        self.starting_distribution_kernel = self.add_weight(
            shape=[1, self.config.heads, self.num_states],
            initializer="zeros" if self.config.starting_distribution_trainable
                else tf.keras.initializers.Constant(
                    tf.expand_dims([[
                            3., -1., -1., -1., 1., 1.5, 1.,
                            -2., -2., -2., -2., -2., -2., -2., -2.
                        ]] * self.config.heads,
                        axis=0,
                    )
                ),
            name="starting_distribution_kernel",
            trainable=self.config.starting_distribution_trainable,
        )
        self.built = True

    def make_transition_init(self, k: int = 1, sd: float = 0.05) -> np.ndarray:
        init = []
        for edge in self.indices:
            if is_intergenic_loop(edge):
                p_loop = 1 - 1 / self.config.initial_ir_len
                init.append(-np.log(1/p_loop - 1))
            elif is_intron_loop(edge, k):
                p_loop = 1 - 1 / self.config.initial_intron_len
                init.append(-np.log(1/p_loop - 1))
            elif is_exon_transition(edge, k):
                p_next_exon = 1 - 1 / self.config.initial_exon_len
                init.append(-np.log(1/p_next_exon - 1))
            elif is_exon_1_out_transition(edge, k):
                init.append(np.log(1/2))
            elif is_intergenic_out_transition(edge, k):
                init.append(np.log(1/k) + np.random.normal(0, sd))
            else:
                init.append(0)
        return np.array(init).reshape(self.config.heads, self.num_transitions)

    def make_transition_indices(self) -> np.ndarray:
        IR = 0
        I = list(range(1, 4))
        E = list(range(4, 7))
        START = 7
        EI = list(range(8, 11))
        IE = list(range(11, 14))
        STOP = 14
        indices = [
            (IR, IR), (IR, START), (STOP, IR), (START, E[1]), (E[1], STOP),
        ]
        for cds in range(3):
            indices.append((E[cds], E[(cds+1) % 3]))
            indices.append((E[cds], EI[cds]))
            indices.append((EI[cds], I[cds]))
            indices.append((I[cds], I[cds]))
            indices.append((I[cds], IE[cds]))
            indices.append((IE[cds], E[cds]))

        repeats = np.arange(self.config.heads).reshape(self.config.heads, 1, 1)
        repeats = np.tile(repeats, (1, len(indices), 1))
        indices = np.tile(indices, (self.config.heads, 1, 1))
        indices = np.concatenate([repeats, indices], axis=-1, dtype=np.int64)
        return indices.reshape(-1, 3)

    def recurrent_init(self) -> None:
        self.A = self.make_A()
        self.A_transposed = tf.transpose(self.A, (0, 2, 1))

    def make_A_sparse(self, values=None) -> tf.SparseTensor:
        if values is None:
            values = tf.reshape(self.transition_kernel, [-1])
        tensor = tf.SparseTensor(
            indices=self.indices,
            values=values,
            dense_shape=(self.config.heads, self.num_states, self.num_states),
        )
        tensor = tf.sparse.reorder(tensor)
        tensor = tf.sparse.softmax(tensor)
        return tensor

    def make_A(self) -> tf.Tensor:
        A_sparse = self.make_A_sparse()
        A = tf.sparse.to_dense(self.make_A_sparse())
        return A

    def make_log_A(self) -> tf.Tensor:
        A_sparse = self.make_A_sparse()
        log_A_sparse = tf.sparse.map_values(tf.math.log, A_sparse)
        log_A = tf.sparse.to_dense(log_A_sparse, default_value=-1e3)
        return log_A

    def make_initial_distribution(self) -> tf.Tensor:
        return tf.nn.softmax(self.starting_distribution_kernel)  # type: ignore

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
