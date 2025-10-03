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

    def call(self, p: tf.Tensor) -> tf.Tensor:
        entropy = -tf.reduce_sum(
            p * tf.math.log(p + 1e-8),  # type: ignore
            axis=-1,
        )
        loss = tf.reduce_mean(
            entropy * (1.0 - p[..., self.config.class_index])  # type: ignore
        )
        self.add_loss(self.config.weight * loss)
        return p


class RepeatsNonCodingRegularizerConfig(ModelConfig):

    weight: float
    use_reverse_strand: bool
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
