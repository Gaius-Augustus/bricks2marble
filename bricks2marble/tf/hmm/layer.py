import tensorflow as tf
from learnMSA.msa_hmm.MsaHmmCell import HmmCell
from learnMSA.msa_hmm.MsaHmmLayer import MsaHmmLayer as HmmLayer
from learnMSA.msa_hmm.Viterbi import viterbi

from ..config import ModelConfig, with_config
from .emitter import Emitter
from .transitioner import Transitioner


class HMMLayerConfig(ModelConfig):

    start_codons: list[tuple[str, float]] = [("ATG", 1.)]
    stop_codons: list[tuple[str, float]] = [
        ("TAG", .34), ("TAA", .33), ("TGA", .33),
    ]
    intron_begin_pattern: list[tuple[str, float]] = [("NGT", 1.)]
    intron_end_pattern: list[tuple[str, float]] = [("AGN", 1.)]

    heads: int = 1
    use_reverse_strand: bool = False
    parallel_factor: int = 1

    initial_exon_len: int = 100
    initial_intron_len: int = 10000
    initial_ir_len: int = 10000
    train_transitions: bool = True
    train_start_dist: bool = True
    share_noncoding_params: bool = False


@with_config(HMMLayerConfig)
class HMMLayer(HmmLayer):

    config: HMMLayerConfig

    def post_config_init(self) -> None:
        super().__init__(
            cell=None,
            num_seqs=None,
            use_prior=False,
            parallel_factor=self.config.parallel_factor,
        )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        emitter = Emitter(
            start_codons=self.config.start_codons,
            stop_codons=self.config.stop_codons,
            intron_begin_pattern=self.config.intron_begin_pattern,
            intron_end_pattern=self.config.intron_end_pattern,
            heads=self.config.heads,
            use_reverse_strand=self.config.use_reverse_strand,
            share_noncoding_params=self.config.share_noncoding_params,
        )
        transitioner = Transitioner(
            heads=self.config.heads,
            initial_exon_len=self.config.initial_exon_len,
            initial_intron_len=self.config.initial_intron_len,
            initial_ir_len=self.config.initial_ir_len,
            starting_distribution_trainable=self.config.train_start_dist,
            transitions_trainable=self.config.train_transitions,
        )

        self.cell = HmmCell(
            [emitter.config.n_states] * (self.config.heads),
            dim=input_shape[-1],
            emitter=emitter,
            transitioner=transitioner,
            use_fake_step_counter=True,
            name="gene_pred_hmm_cell",
        )
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
        training: bool = False,
        use_loglik: bool = True,
    ) -> tf.Tensor:
        if self.config.use_reverse_strand:
            B, T, D = tf.unstack(tf.shape(nuc))  # type: ignore
            nuc = tf.expand_dims(nuc, 0)
            nuc_reverse = tf.gather(nuc, [3, 2, 1, 0, 4], axis=-1)
            nuc_reverse = tf.reverse(nuc_reverse, [-2])
            nuc = tf.concat((nuc, nuc_reverse), axis=0)  # type: ignore
            nuc = tf.reshape(nuc, (1, 2*B, T, D))
            x = tf.expand_dims(x, 0)
            x = tf.concat((x, tf.reverse(x, [-2])), axis=0)  # type: ignore
            x = tf.reshape(x, (1, 2*B, T, -1))
        else:
            nuc = tf.expand_dims(nuc, 0)
            x = tf.expand_dims(x, 0)
        x = tf.concat((x, nuc), axis=-1)  # type: ignore

        x, _, _ = self.state_posterior_log_probs(
            x,
            return_prior=True,
            training=training,
            no_loglik=not use_loglik,
        )  # type: ignore
        x = tf.transpose(x, [1, 2, 0, 3])

        if self.config.use_reverse_strand:
            # 2*B, T, H, D
            x = tf.reshape(x, tf.concat((
                (2, ),
                (tf.shape(x)[0]//2, ),  # type: ignore
                tf.shape(x)[1:]  # type: ignore
            ), 0))
            # 2, B, T, H, D
            x = tf.concat(
                (x[0:1], tf.reverse(x[1:2], [-3])  # type: ignore
            ), 0)
            x = tf.transpose(x, [1, 2, 0, 3, 4])
            # B, T, 2, H, D
            x = tf.reshape(x, tf.concat((
                tf.shape(x)[:2],  # type: ignore
                (tf.shape(x)[2]*tf.shape(x)[3], ),  # type: ignore
                tf.shape(x)[4:],  # type: ignore
            ), 0))
            # B, T, 2*H, D

        return x

    def viterbi(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
    ) -> tf.Tensor:
        self.cell.recurrent_init()
        if self.config.use_reverse_strand:
            B, T, D = tf.unstack(tf.shape(nuc))  # type: ignore
            nuc = tf.expand_dims(nuc, 0)
            nuc_reverse = tf.gather(nuc, [3, 2, 1, 0, 4], axis=-1)
            nuc_reverse = tf.reverse(nuc_reverse, [-2])
            nuc = tf.concat((nuc, nuc_reverse), axis=0)  # type: ignore
            nuc = tf.reshape(nuc, (1, 2*B, T, D))
            x = tf.expand_dims(x, 0)
            x = tf.concat((x, tf.reverse(x, [-2])), axis=0)  # type: ignore
            x = tf.reshape(x, (1, 2*B, T, -1))
        else:
            nuc = tf.expand_dims(nuc, 0)
            x = tf.expand_dims(x, 0)
        x = tf.concat((x, nuc), axis=-1)  # type: ignore

        x = viterbi(
            x,
            self.cell,
            parallel_factor=self.parallel_factor,
        )  # type: ignore
        x = tf.transpose(x, [1, 2, 0])

        if self.config.use_reverse_strand:
            # 2*B, T, H
            x = tf.reshape(x, tf.concat((
                (2, ),
                (tf.shape(x)[0]//2, ),  # type: ignore
                tf.shape(x)[1:]  # type: ignore
            ), 0))
            # 2, B, T, H
            x = tf.concat(
                (x[0:1], tf.reverse(x[1:2], [-2])  # type: ignore
            ), 0)
            x = tf.transpose(x, [1, 2, 0, 3])
            # B, T, 2, H
            x = tf.reshape(x, tf.concat((
                tf.shape(x)[:2],  # type: ignore
                (tf.shape(x)[2]*tf.shape(x)[3], ),  # type: ignore
            ), 0))
            # B, T, 2*H

        return x
