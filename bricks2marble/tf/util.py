import tensorflow as tf

from .config import ModelConfig, with_config


class UncertainPredictionRegularizerConfig(ModelConfig):

    class_index: int
    weight: float


@with_config(UncertainPredictionRegularizerConfig)
class UncertainPredictionRegularizer(tf.keras.layers.Layer):

    config: UncertainPredictionRegularizerConfig

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


@tf.function
def shared_sparse_tensor(
    indices: tf.Tensor,
    values: tf.Tensor,
    shape: tf.Tensor,
    share: tf.Tensor | None = None,
) -> tf.SparseTensor:
    """Creates a sparse tensor with optional parameter sharing.

    Args:
        indices (tf.Tensor): Tensor of shape ``(N, 2)``, dtype int64.
        values (tf.Tensor): Tensor of shape ``(K, )``, dtype float32.
        shape (tf.Tensor): Shape of the resulting tensor.
        share (tf.Tensor, optional): Tensor of shape ``(K, 2)``, dtype
            int32 or int64. Defaults to no parameter sharing.

    Returns:
        tf.Tensor: A sparse tensor with shared parameters placed at
            given indices.
    """
    indices = tf.convert_to_tensor(indices, dtype=tf.int64)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    shape = tf.convert_to_tensor(shape, dtype=tf.int64)

    N = tf.shape(indices, out_type=tf.int64)[0]  # type: ignore
    value_indices = tf.range(N, dtype=tf.int64)
    if share is not None:
        share = tf.convert_to_tensor(share, dtype=tf.int64)
        total_count = tf.constant(0, dtype=tf.int64)
        for i in tf.range(tf.shape(share)[0], dtype=tf.int64):  # type: ignore
            start = share[i, 0]  # type: ignore
            end = share[i, 1]  # type: ignore
            count = end - start

            tail = tf.slice(value_indices, [start], [N - start])
            rolled_tail = tf.roll(tail, shift=(count - 1), axis=0)
            update_tail_indices = tf.range(start, N, dtype=tf.int64)
            value_indices = tf.tensor_scatter_nd_update(
                value_indices,
                tf.expand_dims(update_tail_indices, 1),
                rolled_tail,
            )

            shared_idx = tf.range(start, end, dtype=tf.int64)
            value_indices = tf.tensor_scatter_nd_update(
                value_indices,
                tf.expand_dims(shared_idx, 1),
                tf.fill([count], start-total_count),
            )
            total_count += count - 1

    final_values = tf.gather(values, value_indices)
    sparse_tensor = tf.SparseTensor(
        indices=indices,
        values=final_values,
        dense_shape=shape,
    )
    return tf.sparse.reorder(sparse_tensor)
