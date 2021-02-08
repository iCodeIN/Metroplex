# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Maintain moving averages of parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
import tensorflow.compat.v1 as tf

def is0(x):
  return tf.math.equal(x, 0)

def istrue(x):
  return not is0(x)

def assign(variable, value, name=None):

  with ops.name_scope(name, "Assign",
                      [variable, value]) as scope:

    def update_fn(v, value):
      return state_ops.assign(v, value, name=scope)

    def update(strategy, v, value):
      return strategy.extended.update(v, update_fn, args=(value,))

    replica_context = distribution_strategy_context.get_replica_context()
    if replica_context:
      # In a replica context, we update variable using the mean of value across
      # replicas.
      def merge_fn(strategy, v, value):
        value = strategy.extended.reduce_to(ds_reduce_util.ReduceOp.MEAN, value, v)
        return update(strategy, v, value)

      return replica_context.merge_call(merge_fn, args=(variable, value))
    else:
      strategy = distribution_strategy_context.get_cross_replica_context()
      return update(strategy, variable, value)


class Stabilizer(object):

  def __init__(self, params, name="Stabilizer"):

    self._name = name
    self._averages = {}

    self.dtype = tf.float32
    self.params = params
    dtype = self.dtype
    self.save_freq = params["save_freq"]
    assert params["steps_per_checkpoint"] % self.save_freq == 0
    blacklist = ['global_step:0']
    self.variables = [var for var in tf.global_variables() if var.name not in blacklist]

  @property
  def name(self):
    return self._name

  def var_creation(self):

    for var in self.variables:
      if var.dtype.base_dtype not in [
          dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
      ]:
        raise TypeError("The variables must be half, float, or double: %s" %
                        var.name)

      if var.experimental_ref() not in self._averages:
        with ops.init_scope():
          if isinstance(var, variables.Variable):
            avg = slot_creator.create_slot(
                var,
                var.initialized_value(),
                self.name,
                colocate_with_primary=True)
            ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
          else:
            avg = slot_creator.create_zeros_slot(
                var,
                self.name,
                colocate_with_primary=(var.op.type in [
                    "Variable", "VariableV2", "VarHandleOp"
                ]))
        self._averages[var.experimental_ref()] = avg

  def duplicate(self, mode):
    # copy the model & optimizer states
    with ops.name_scope(mode) as scope:
      updates = []
      for var in self.variables:
        if mode == 'save':
          write = self._averages[var.experimental_ref()]
          read = var
        else:
          write = var
          read = self._averages[var.experimental_ref()]
        updates.append(assign(write, read))
      return control_flow_ops.group(*updates, name=scope)

  def apply(self, stats, gs):
    self.var_creation()
    global_step = gs - 1
    loss = stats['elbo']
    def tensor(value):
      return tf.constant(value, dtype=self.dtype)
    duplicate_mode = tensor(0)
    if is0(global_step):
      duplicate_mode = tensor(1)
    elif is0((global_step+1) % self.save_freq):
      if tf.math.is_nan(loss):
        # no non-trainable parameters in the model, so this works
        duplicate_mode = tensor(2)
      else:
        duplicate_mode = tensor(1)     

    train_ops = [tf.switch_case(tf.cast(duplicate_mode, tf.int32), 
                      branch_fns={0: tf.no_op, 1: (lambda: self.duplicate(mode='save')), 
                                  2: (lambda: self.duplicate(mode='restore'))})]
    stats.update(st=global_step, duplicate_mode=duplicate_mode)
    return tf.group(train_ops), stats


  def average(self, var):
    return self._averages.get(var.experimental_ref(), None)

  def average_name(self, var):
    if var.experimental_ref() in self._averages:
      return self._averages[var.experimental_ref()].op.name
    return ops.get_default_graph().unique_name(
        var.op.name + "/" + self.name, mark_as_used=False)

  def variables_to_restore(self, moving_avg_variables=None):
    name_map = {}
    if moving_avg_variables is None:
      # Include trainable variables and variables which have been explicitly
      # added to the moving_average_variables collection.
      moving_avg_variables = variables.trainable_variables()
      moving_avg_variables += variables.moving_average_variables()
    # Remove duplicates
    moving_avg_variables = set(moving_avg_variables)
    # Collect all the variables with moving average,
    for v in moving_avg_variables:
      name_map[self.average_name(v)] = v
    # Make sure we restore variables without moving averages as well.
    moving_avg_variable_names = set([v.name for v in moving_avg_variables])
    for v in list(set(variables.global_variables())):
      if v.name not in moving_avg_variable_names and v.op.name not in name_map:
        name_map[v.op.name] = v
    return name_map



'''
def __init__(self, params, name="Stabilizer"):

    self._name = name
    self._averages = {}

    self.dtype = tf.float32
    self.params = params
    dtype = self.dtype
    self.num_buckets = 100
    self.save_freq = params["save_freq"]
    assert params["steps_per_checkpoint"] % self.save_freq == 0
    self.decay = 0.99
    blacklist = ['global_step:0']
    self.variables = [var for var in tf.global_variables() if var.name not in blacklist]
    zero = tf.zeros_initializer(dtype)
    # take care of the initialization & checkpointing
    self.loss_ema = tf.get_variable("loss_ema", [], initializer=zero, dtype=dtype, trainable=False)
    self.repeat_count = tf.get_variable("repeat_count", [], initializer=zero, dtype=dtype, trainable=False)
    self.losses = tf.get_variable("losses", [self.save_freq], initializer=zero, dtype=dtype, trainable=False)
    self.tmp_loss_ema = tf.get_variable("tmp_loss_ema", [], initializer=zero, dtype=dtype, trainable=False) 
    self.tmp_loss_sum = tf.get_variable("tmp_loss_sum", [], initializer=zero, dtype=dtype, trainable=False)    
    self.deviations = tf.get_variable("deviations", [self.num_buckets], initializer=zero, dtype=dtype, trainable=False)


  @property
  def name(self):
    return self._name

  def var_creation(self):

    for var in self.variables:
      if var.dtype.base_dtype not in [
          dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
      ]:
        raise TypeError("The variables must be half, float, or double: %s" %
                        var.name)

      if var.experimental_ref() not in self._averages:
        with ops.init_scope():
          if isinstance(var, variables.Variable):
            avg = slot_creator.create_slot(
                var,
                var.initialized_value(),
                self.name,
                colocate_with_primary=True)
            ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
          else:
            avg = slot_creator.create_zeros_slot(
                var,
                self.name,
                colocate_with_primary=(var.op.type in [
                    "Variable", "VariableV2", "VarHandleOp"
                ]))
        self._averages[var.experimental_ref()] = avg

  def duplicate(self, mode):
    # copy the model & optimizer states
    with ops.name_scope(mode) as scope:
      decay = ops.convert_to_tensor(self.decay, name="decay")
      updates = []
      for var in self.variables:
        if mode == 'save':
          write = self._averages[var.experimental_ref()]
          read = var
        else:
          write = var
          read = self._averages[var.experimental_ref()]
        updates.append(assign(write, read))
      return control_flow_ops.group(*updates, name=scope)

  def apply(self, stats, gs):
    self.var_creation()
    global_step = gs - 1
    train_op = []
    loss = stats['elbo']
    def tensor(value):
      if isinstance(value, bool):
        value = 1 if value else 0
      return tf.constant(value, dtype=self.dtype)
    duplicate_mode = tensor(0)
    train_op += [self.losses[global_step % self.save_freq].assign(loss)] #
    train_op += [self.tmp_loss_ema.assign(self.tmp_loss_ema * self.decay + self.loss * (1 - self.decay))]
    with tf.control_dependencies(train_op):
      condition = tensor(False)
      deviation_update = tensor(False)
      cur_dev = tf.zeros(1, dtype=self.dtype)
      saved_steps = global_step // self.save_freq
      loss_ema, repeat_count = map(lambda x: tf.identity(x), (self.loss_ema, self.repeat_count))
      if is0(self.loss_ema):
        duplicate_mode = tensor(1)
        loss_ema = loss # check if checkpointing interferes
      elif is0((global_step+1) % self.save_freq):
        cur_dev = tf.math.reduce_prod(self.losses / loss_ema) #
        nonzeros = tf.reduce_sum(1 - tf.cast(is0(self.deviations), self.dtype))
        k = tf.cast(nonzeros * 0.1, tf.int32)
        threshold = tf.math.top_k(self.deviations, k=k, sorted=True)[0][-1] if k > 0 else tf.constant(1000, dtype=self.dtype)
        if tf.math.is_nan(cur_dev):
          condition = tensor(True)
          repeat_count = tensor(0)
        elif cur_dev > threshold:
          if repeat_count < 1:
            condition = tensor(True)
            repeat_count = repeat_count + 1
          else:
            condition = tensor(False)
            repeat_count = tensor(0)
        if istrue(condition):        
          # no non-trainable parameters in the model, so this works
          duplicate_mode = tensor(2)
        else:
          deviation_update = tensor(True)
          duplicate_mode = tensor(1)
          x = loss_ema
          for idx in range(self.save_freq):
            x = x * self.decay + self.losses[idx] * (1 - self.decay) #
          loss_ema = x       

      train_ops = [tf.switch_case(tf.cast(duplicate_mode, tf.int32), 
                        branch_fns={0: tf.no_op, 1: (lambda: self.duplicate(mode='save')), 
                                    2: (lambda: self.duplicate(mode='restore'))})]
      for tensor, var in [(loss_ema, self.loss_ema), (repeat_count, self.repeat_count)]:
        train_ops += [var.assign(tensor)]        
      train_ops += [tf.cond(istrue(deviation_update), 
                             true_fn=lambda: self.deviations[saved_steps % self.num_buckets].assign(cur_dev),
                             false_fn=lambda: tf.identity(self.deviations[saved_steps % self.num_buckets]))]
      #if istrue(deviation_update):
      #  deviation_op = self.deviations[saved_steps % self.num_buckets].assign(cur_dev)       
      #else:
      #  deviation_op = tf.no_op
      #train_ops += [deviation_op]
      stats.update(st=global_step, returning=condition, deviation_update=deviation_update, loss_ema=self.loss_ema, repeat_count=self.repeat_count)
      return tf.group(train_ops), stats
'''