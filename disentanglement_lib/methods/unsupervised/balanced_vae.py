# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Library of losses for disentanglement learning.

Implementation of VAE based models for unsupervised learning of disentangled
representations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
import tensorflow as tf
import gin.tf


def compute_gaussian_kl_with_parts(z_mean, z_logvar):
  """Compute KL divergence between input Gaussian and Standard Normal."""
  return (tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean), [1]), name="size_loss"),
    tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(z_logvar) - z_logvar - 1, [1]), name="variance_loss"))

class BalancedBaseVAE(gaussian_encoder_model.GaussianEncoderModel):
  """Abstract base class of a basic Gaussian encoder model."""

  def model_fn(self, features, labels, mode, params):
    """TPUEstimator compatible model function."""
    del labels
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    data_shape = features.get_shape().as_list()[1:]
    z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
    z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
    reconstructions = self.decode(z_sampled, data_shape, is_training)
    per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
    reconstruction_loss = tf.reduce_mean(per_sample_loss)
    size_loss, variance_loss = compute_gaussian_kl_with_parts(z_mean, z_logvar)
    kl_loss = size_loss + variance_loss
    regularizer = self.regularizer(size_loss, variance_loss, z_mean, z_logvar, z_sampled)
    loss = tf.add(reconstruction_loss, regularizer, name="loss")
    elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = optimizers.make_vae_optimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
      train_op = tf.group([train_op, update_ops])
      tf.summary.scalar("reconstruction_loss", reconstruction_loss)
      tf.summary.scalar("elbo", -elbo)
      tf.summary.scalar("regularizer_loss", regularizer)
      
      logging_hook = tf.train.LoggingTensorHook({
          "loss": loss,
          "reconstruction_loss": reconstruction_loss,
          "elbo": -elbo,
          "size_loss": size_loss,
          "variance_loss": variance_loss,
          "regularizer_loss": regularizer
      },
                                                every_n_iter=100)
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                       "regularizer", "kl_loss" ,"size_loss", "variance_loss", "regularizer_loss"),
                        [reconstruction_loss, -elbo, regularizer, kl_loss, size_loss, variance_loss, regularizer_loss]))
    else:
      raise NotImplementedError("Eval mode not supported.")

  def gaussian_encoder(self, input_tensor, is_training):
    """Applies the Gaussian encoder to images.

    Args:
      input_tensor: Tensor with the observations to be encoded.
      is_training: Boolean indicating whether in training mode.

    Returns:
      Tuple of tensors with the mean and log variance of the Gaussian encoder.
    """
    return architectures.make_gaussian_encoder(
        input_tensor, is_training=is_training)

  def decode(self, latent_tensor, observation_shape, is_training):
    """Decodes the latent_tensor to an observation."""
    return architectures.make_decoder(
        latent_tensor, observation_shape, is_training=is_training)


def make_metric_fn(*names):
  """Utility function to report tf.metrics in model functions."""

  def metric_fn(*args):
    return {name: tf.metrics.mean(vec) for name, vec in zip(names, args)}

  return metric_fn


@gin.configurable("balanced_beta_vae")
class BalancedBetaVAE(BalancedBaseVAE):
  """BVAE model."""

  def __init__(self, beta_size=gin.REQUIRED, beta_variance=gin.REQUIRED):
    """Creates a beta-VAE model with different weights for size and variance losses.

    Args:
      beta_size: Hyperparameter for the regularizer of the size loss.
      beta_variance: Hyperparameter for the regularizer of the variance loss.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    self.beta_size = beta_size
    self.beta_variance = beta_variance

  def regularizer(self, size_loss, variance_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta_size * size_loss + self.beta_variance * variance_loss


@gin.configurable("augmented_variance_vae")
class AugmentedVarianceVAE(BalancedBaseVAE):
  """VAE model with augmented variance loss."""

  def __init__(self, mean_var_weight=gin.REQUIRED, variance_weight=gin.REQUIRED):
    """Creates a VAE model with a regularizer where the variance of means is incorporated.

    Args:
      mean_var_weight: Weight for the variance of means component.
      variance_weight: Weight for the variance component.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    self.mean_var_weight = mean_var_weight
    self.variance_weight = variance_weight

  def regularizer(self, size_loss, variance_loss, z_mean, z_logvar, z_sampled):
        variance = tf.exp(z_logvar)

        m = tf.reduce_mean(z_mean, axis=0, keepdims=True)
        devs_squared = tf.square(z_mean - m)
        mean_variance = tf.reduce_mean(devs_squared, axis=0, keepdims=False)

        total_variance = self.variance_weight * variance + self.mean_var_weight * mean_variance
        loss = 0.5 * tf.reduce_sum(-1 - tf.log(total_variance) + total_variance, [1])
        return tf.reduce_mean(loss, name="augmented_variance_loss")




