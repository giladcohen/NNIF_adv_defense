from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

class ReduceLROnPlateau(object):
  """Reduce learning rate when a metric has stopped improving.
  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.
  Example:
  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```
  Arguments:
      factor: factor by which the learning rate will
          be reduced. new_lr = lr * factor
      patience: number of epochs with no improvement
          after which learning rate will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of {min, max}. In `min` mode,
          lr will be reduced when the quantity
          monitored has stopped decreasing; in `max`
          mode it will be reduced when the quantity
          monitored has stopped increasing.
      min_delta: threshold for measuring the new optimum,
          to only focus on significant changes.
      cooldown: number of epochs to wait before resuming
          normal operation after lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               factor=0.9,
               patience=3,
               verbose=1,
               mode='max',
               min_delta=1e-4,
               cooldown=2,
               init_lr=0.1,
               min_lr=0):
    super(ReduceLROnPlateau, self).__init__()

    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.init_lr = init_lr
    self.curr_lr = init_lr
    self.mode = mode
    self.monitor_op = None
    self.new_best = False
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    if self.mode == 'min':
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.Inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.Inf
    self.cooldown_counter = 0
    self.wait = 0
    self.curr_lr = self.init_lr
    self.new_best = False

  def on_train_begin(self):
    self._reset()

  def on_epoch_end(self, epoch, metric):
    self.new_best = False

    if self.in_cooldown():
      self.cooldown_counter -= 1
      self.wait = 0

    if self.monitor_op(metric, self.best):
      self.best = metric
      self.wait = 0
      self.new_best = True
    elif not self.in_cooldown():
      self.wait += 1
      if self.wait >= self.patience:
        if self.curr_lr > self.min_lr:
          new_lr = self.curr_lr * self.factor
          new_lr = max(new_lr, self.min_lr)
          if self.verbose > 0:
            print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate from %s to %s.' % (epoch + 1, self.curr_lr, new_lr))
          self.curr_lr = new_lr
          self.cooldown_counter = self.cooldown + 1
          self.wait = 0

  def in_cooldown(self):
    return self.cooldown_counter > 0

  def get_curr_lr(self):
    return self.curr_lr

  def was_improvement(self):
    return self.new_best
