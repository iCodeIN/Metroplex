from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_function


def get_cpu_stats_over_ranks(stat_dict):
    keys = sorted(stat_dict.keys())
    messages = tf.stack([tf.reshape(tf.cast(stat_dict[k], tf.float32), []) for k in keys])
    allreduced = tf.tpu.cross_replica_sum(messages) / tpu_function.get_tpu_context().number_of_shards      
    return {k: allreduced[i] for (i, k) in enumerate(keys)}

def get_optimizer(stats, params):
    """Creates and returns an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    # set defaults
    end_step = params.get("lr_decay_end", params["train_steps"])
    lr_decay = params.get("lr_decay", "cosine")
    warmup_steps = params.get("warmup_steps", 3000)
    gradient_clipping = params.get("gradient_clipping", 1.0)
    optimizer_name = params.get("optimizer", "adam")
    dtype = tf.float32

    learning_rate = tf.constant(value=params["lr"], shape=[], dtype=dtype)
    clip_value = tf.constant(gradient_clipping, dtype=dtype)

    if lr_decay == "linear":
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            end_step,
            end_learning_rate=params["lr"] * 0.1,  # Decrease to 10% of initial LR according to GPT-3 paper
            power=1.0,
            cycle=False)
    elif lr_decay == "cosine":
        learning_rate = tf.train.cosine_decay(
            learning_rate,
            global_step,
            end_step,
            alpha=0.1  # Alpha is min lr value as a fraction of init lr.
        )

    if warmup_steps > 0:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, dtype)
        warmup_steps_float = tf.cast(warmup_steps_int, dtype)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, dtype)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    if optimizer_name.lower() == "adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=params.get("beta_1", 0.9),
            beta2=params.get("beta_2", 0.999),
            epsilon=params.get("epsilon", 1e-6),
        )
    elif optimizer_name.lower() == "adafactor":
        # need to modify this one
        optimizer = tf.optimize.AdafactorOptimizer(
            learning_rate=learning_rate,
            decay_rate=params.get("weight_decay", 0.0),
            beta1=params.get("beta_1", 0.9),
            epsilon1=params.get("epsilon_1", 1e-30),
            epsilon2=params.get("epsilon_2", 1e-3)
        )
    else:
        raise ValueError(f"{optimizer_name} not recognized")

    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    grads_vars = optimizer.compute_gradients(stats['elbo'])
    grads, variables = zip(*grads_vars)
    #grads, variables = list(grads), list(variables)

    grads_fp = [tf.cast(grad, dtype) for grad in grads]
    grads_fp, grad_norm = tf.clip_by_global_norm(grads_fp, clip_value)
    grad_norm *= tpu_function.get_tpu_context().number_of_shards 
    grads = [tf.cast(grad, variables[0].dtype) for grad in grads_fp]

    zero = tf.zeros(1, dtype=dtype)
    one = tf.ones(1, dtype=dtype)
    is0 = lambda x: tf.math.equal(x, 0)
    zero_or_one = lambda x: zero if is0(x) else one 
    nans_f = lambda name: zero_or_one(tf.reduce_sum(tf.cast(tf.math.is_nan(stats[name]), dtype)))
    distortion_nans, rate_nans = map(nans_f, ('distortion', 'rate'))
    stats.update(dict(rate_nans=rate_nans, distortion_nans=distortion_nans, grad_norm=grad_norm))
    stats = get_cpu_stats_over_ranks(stats)
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if is0(stats['distortion_nans']) and is0(stats['rate_nans']) and (grad_norm < params["skip_threshold"]):
        train_ops = optimizer.apply_gradients(list(zip(grads, variables)), global_step=global_step)
        skipped_updates = zero
    else:
        train_ops = tf.group([tf.assign_add(global_step, 1)])
        skipped_updates = one
    stats.update(skipped_updates=skipped_updates) 
    return train_ops, stats, global_step