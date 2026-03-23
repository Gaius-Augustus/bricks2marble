from collections.abc import Callable

import numpy as np
import tensorflow as tf
from hidten.tf import TFTransitioner
from hidten.tf.util import T_TFTensor


class GeneTransitioner(TFTransitioner):

    _non_trainable_rows = np.array([
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])

    def __init__(
        self,
        initial_exon_len: float,
        initial_intron_len: float,
        initial_ir_len: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.exon = tf.math.log(initial_exon_len - 1)
        self.intron = tf.math.log(initial_intron_len - 1)
        self.ir = tf.math.log(initial_ir_len - 1)

    def build(self, input_shape=None) -> None:
        super().build(input_shape)
        self.kernel.trainable = False
        self.kernel3 = self.add_weight(
            shape=(3, ),
            initializer=tf.keras.initializers.Constant(
                [self.ir, self.intron, self.exon],
            ),
            name="kernel3",
            trainable=self.train_transitions,
        )

    def _build_matrix(
        self,
        matrix_activation: Callable[[T_TFTensor], T_TFTensor],
        delta: T_TFTensor | None = None,
    ) -> T_TFTensor:
        p = tf.sigmoid(
            self.kernel3 + delta if delta is not None else self.kernel3
        )
        A = tf.concat(([
            p[0], 0, 0, 0, 0, 0, 0, 1-p[0], 0, 0, 0, 0, 0, 0, 0,
            0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0,
            0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0,
            0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0,
            0, 0, 0, 0, 0, p[2], 0, 0, 1-p[2], 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, p[2], 0, 0, (1-p[2])/2, 0, 0, 0, 0, (1-p[2])/2,
            0, 0, 0, 0, p[2], 0, 0, 0, 0, 0, 1-p[2], 0, 0, 0, 0,
        ], self._non_trainable_rows), axis=0)
        A = tf.tile(
            tf.expand_dims(tf.reshape(A, (15, 15)), 0),
            [self.heads, 1, 1],
        )
        return A
