import tensorflow as tf
from hidten.config import ModelConfig, with_config


class UncertainPredictionRegularizerConfig(ModelConfig):

    class_index: int
    weight: float


@with_config(UncertainPredictionRegularizerConfig)
class UncertainPredictionRegularizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = UncertainPredictionRegularizerConfig(**kwargs)
        self.nudge_weight = tf.Variable(
            self.config.weight,
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, p: tf.Tensor) -> tf.Tensor:
        entropy = -tf.reduce_sum(
            p * tf.math.log(p + 1e-8),  # type: ignore
            axis=-1,
        )
        loss = tf.reduce_mean(
            entropy * (1.0 - p[..., self.config.class_index])  # type: ignore
        )
        self.add_loss(self.nudge_weight * loss)
        return p


class RepeatsNonCodingRegularizerConfig(ModelConfig):

    weight: float
    coding_start_index: int


@with_config(RepeatsNonCodingRegularizerConfig)
class RepeatsNonCodingRegularizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = RepeatsNonCodingRegularizerConfig(**kwargs)
        self.repeats_noncoding_weight = tf.Variable(
            self.config.weight,
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, p: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
        # p: (B, T, 2*H(H), D) | r: (B, T, 1)
        pnc = tf.reduce_sum(
            p[:, :, :, self.config.coding_start_index:],
            axis=-1,
        )
        loss = tf.reduce_mean(tf.reduce_sum(r * pnc, axis=-1))
        self.add_loss(self.repeats_noncoding_weight * loss)
        return p


class IntronParameterRegularizerConfig(ModelConfig):

    weight: float
    intron_state_chain: int
    start_value: list[float]


@with_config(IntronParameterRegularizerConfig)
class IntronParameterRegularizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = IntronParameterRegularizerConfig(**kwargs)
        self.intron_weight = tf.Variable(
            self.config.weight,
            trainable=False,
            dtype=tf.float32,
        )
        self.start_value = tf.Variable(
            self.config.start_value,
            trainable=False,
            dtype=tf.float32,
        )
        self.heads = len(self.config.start_value)

    def call(self, A: tf.Tensor) -> tf.Tensor:
        diff = tf.norm(self.start_value - tf.concat([
            tf.reduce_mean(tf.linalg.diag_part(A[i])[
                1:1+3*self.config.intron_state_chain
            ], keepdims=True)
            for i in range(self.heads)
        ], axis=0), ord=1)
        loss = self.intron_weight * diff
        self.add_loss(loss)
        return A
