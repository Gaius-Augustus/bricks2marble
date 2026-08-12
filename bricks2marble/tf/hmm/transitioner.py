from collections.abc import Callable

import tensorflow as tf
from hidten.tf import TFTransitioner
from hidten.tf.util import T_TFTensor


class GeneTransitioner(TFTransitioner):

    def __init__(
        self,
        initial_exon_len: float,
        initial_intron_len: float,
        initial_ir_len: float,
        no_spliced_stop: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.exon = tf.math.log(initial_exon_len - 1)
        self.intron = tf.math.log(initial_intron_len - 1)
        self.ir = tf.math.log(initial_ir_len - 1)
        self.no_spliced_stop = no_spliced_stop

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
        if not self.no_spliced_stop:
            self._non_trainable_rows = tf.convert_to_tensor([
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ], tf.float32)
        else:
            self._non_trainable_rows = tf.convert_to_tensor([
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ], tf.float32)

    def _build_matrix(
        self,
        matrix_activation: Callable[[T_TFTensor], T_TFTensor],
        delta: T_TFTensor | None = None,
    ) -> T_TFTensor:
        p = tf.sigmoid(
            self.kernel3 + delta if delta is not None else self.kernel3
        )
        if not self.no_spliced_stop:
            A = tf.concat((tf.convert_to_tensor([
                p[0], 0, 0, 0, 0, 0, 0, 1-p[0], 0, 0, 0, 0, 0, 0, 0,
                0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0,
                0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0,
                0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0,
                0, 0, 0, 0, 0, p[2], 0, 0, 1-p[2], 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, p[2], 0, 0, (1-p[2])/2, 0, 0, 0, 0, (1-p[2])/2,
                0, 0, 0, 0, p[2], 0, 0, 0, 0, 0, 1-p[2], 0, 0, 0, 0,
            ], tf.float32), self._non_trainable_rows), axis=0)
            A = tf.tile(
                tf.expand_dims(tf.reshape(A, (15, 15)), 0),
                [self.heads, 1, 1],
            )
        else:
            A = tf.concat((tf.convert_to_tensor([
                p[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0, 0, 0, 0,
                0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0, 0, 0,
                0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0, 0,
                0, 0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0, 0,
                0, 0, 0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0, 0,
                0, 0, 0, 0, 0, 0, p[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p[1], 0,
                0, 0, 0, 0, 0, 0, 0, 0, p[2], 0, 0, 1-p[2], 0, 0, 0, 1-p[2], 1-p[2], 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, p[2], 0, 0, (1-p[2])/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1-p[2])/2,
                0, 0, 0, 0, 0, 0, 0, p[2], 0, 0, 0, 0, 0, 1-p[2], 1-p[2], 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ], tf.float32), self._non_trainable_rows), axis=0)
            A = tf.tile(
                tf.expand_dims(tf.reshape(A, (24, 24)), 0),
                [self.heads, 1, 1],
            )
        return A