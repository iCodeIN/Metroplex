import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.tpu import tpu_estimator
from .optimizers import get_optimizer
from .metroplex.models import Metroplex
from .utils import scalar_summary, mode_to_str, create_host_call
from configs.config import Hyperparams
from .stabilizer import Stabilizer

def model_fn(features, labels, mode, params):

    H = W = params["dataset"]["image_size"]  # TODO: check equal
    mode_str = mode_to_str(mode)
    batch_size = params[f"{mode_str}_batch_size"]
    n_channels = params.get("input_channels", 3)

    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)

    @tf.autograph.to_graph
    def inner(features):
        with tf.variable_scope("metroplex"):
            model = Metroplex(Hyperparams(params))
            if params.get("use_bf16", False):
                with tf.tpu.bfloat16_scope():
                    stats = model.forward(tf.cast(features, tf.bfloat16))
            else:
                stats = model.forward(features)

        loss = tf.cast(stats['elbo'], tf.float32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op, stats, gs = get_optimizer(stats, params)
            '''stablizer = Stabilizer(params)
            if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.control_dependencies([train_op]):
                    train_ops, stats = stablizer.apply(stats, gs)'''            
        return stats, loss, train_op, gs
    stats, loss, train_op, gs = inner(features)

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(gs, [1])
    loss_t = tf.reshape(loss, [1])
    keys = sorted(stats.keys())
    stats_t = [tf.reshape(tf.cast(stats[k], tf.float32), [1]) for k in keys]
    #reconstruction = features[:,16,16,0]


    def accumulate_stats(stats):
        z = {}
        for k in keys:
            if k in ['elbo', 'grad_norm']:
                vals = stats[k]
                zero = tf.constant(0, dtype=vals.dtype)
                one = tf.constant(1, dtype=vals.dtype)
                finites = tf2.where(tf.math.is_finite(vals), vals, zero)
                num_finites = tf2.where(tf.math.is_finite(vals), one, zero)
                z[k] = tf.reduce_mean(vals)
                z[k + '_filtered'] = tf.reduce_sum(finites) / tf.reduce_sum(num_finites)
            else:
                z[k] = tf.reduce_mean(stats[k])
        return z

    def host_call_fn(gs, loss, *args):
        gs = gs[0]
        loss = tf.math.reduce_mean(loss)
        stats = {keys[idx]: v for idx, v in enumerate(args)}
        stats = accumulate_stats(stats)
        denormalize = lambda x: (x + 1) / 2

        with tf2.summary.create_file_writer(params['model_path']).as_default():
            tf2.summary.scalar('loss', loss, step=gs)
            #tf2.summary.scalar('input_stat', reconstruction[0], step=gs)
            #tf2.summary.image('input_image', denormalize(input), step=gs)
            #tf2.summary.image('reconstruction_image', denormalize(reconstruction), step=gs)
            for k, v in stats.items():
                tf2.summary.scalar(k, v, step=gs)
            return tf.summary.all_v2_summary_ops()

    def metric_fn(gs, loss, *args):
        gs = gs[0]
        loss = tf.math.reduce_mean(loss)
        stats = {keys[idx]: v for idx, v in enumerate(args)}
        stats = accumulate_stats(stats)
        denormalize = lambda x: (x + 1) / 2

        with tf2.summary.create_file_writer(params['model_path']).as_default():
            loss_op = tf.metrics.mean(loss)
            for k, v in stats.items():
                tf2.summary.scalar(k, v, step=gs)
            #with tf2.summary.record_if(loss_op[0] < tf.constant(1e-9)):
            #    tf2.summary.image('eval/input_image', denormalize(input), step=gs)
            #    tf2.summary.image('eval/reconstruction_image', denormalize(reconstruction), step=gs)

            with tf.control_dependencies(tf.summary.all_v2_summary_ops()):
                dummy_op = tf.no_op()

            return {"_loss": loss_op,
                    "zzz_dummy": (tf.constant(0), dummy_op)}


    host_call = (host_call_fn, [gs_t, loss_t] + stats_t)
    metric = (metric_fn, [gs_t, loss_t] + stats_t)

    return tpu_estimator.TPUEstimatorSpec(
        mode,
        loss=loss,
        host_call=host_call if mode == tf.estimator.ModeKeys.TRAIN else None,
        train_op=train_op,
        eval_metrics=metric)