# Modified from code from T5X (https://github.com/google-research/t5x)

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import einops
import jax
from jax import random
from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from flax.linen.module import Module, compact, merge_param

import numpy as np


from flax.linen import partitioning as nn_partitioning

default_kernel_init = nn.initializers.lecun_normal()

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
Axes = Union[int, Iterable[int]]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0)


def reverse_space_to_depth(
    frames: jnp.ndarray,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1) -> jnp.ndarray:
  """Reverse space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
      frames, 'b h w (dh dw c) -> b (h dh) (w dw) c',
      dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
      frames, 'b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c',
      dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
      'Frames should be of rank 4 (batch, height, width, channels)'
      ' or rank 5 (batch, time, height, width, channels)')


def space_to_depth(
    frames: jnp.ndarray,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1) -> jnp.ndarray:
  """Space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
      frames, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
      dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
      frames, 'b (t dt) (h dh) (w dw) c -> b t (h w) (dt dh dw c)',
      dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
      'Frames should be of rank 4 (batch, height, width, channels)'
      ' or rank 5 (batch, time, height, width, channels)')


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
    'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
    'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
  lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: Initializer = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               abs_bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
      DenseGeneral,
      axis=-1,
      features=(self.num_heads, self.head_dim),
      kernel_axes=('embed', 'joined_kv'),
      dtype=self.dtype)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      cache_mask = self.variable('cache', 'cache_mask', jnp.zeros,
                                 (query.shape[0], 1, 1, query.shape[1]), jnp.float32)
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))

        # Create a OHE of the current index. NOTE: the index is increased below.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
        # In order to update the key, value caches with the current key and
        # value, we move the length axis to the back, similar to what we did for
        # the cached ones above.
        # Note these are currently the key and value of a single position, since
        # we feed one position at a time.
        one_token_key = jnp.moveaxis(key, -3, -1)
        one_token_value = jnp.moveaxis(value, -3, -1)
        # Update key, value caches with our new 1d spatial slices.
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + one_token_key * one_hot_indices
        value = cached_value.value + one_token_value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # Move the keys and values back to their original shapes.
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)

        # Causal mask for cached decoder self-attention: our single query
        # position should only attend to those key positions that have already
        # been generated and cached, not the remaining zero elements.
        # mask = jnp.logical_or(cache_mask.value, mask).astype(jnp.int32)
        # cache_mask.value = mask

        # if cur_index == 20:
        # import ipdb; ipdb.set_trace()

        mask = (cache_mask.value + mask * one_hot_indices).astype(jnp.float32)
        cache_mask.value = mask

        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(length) <= cur_index,
            # (1, 1, length) represent (head dim, query length, key length)
            # query length is 1 because during decoding we deal with one
            # index.
            # The same mask is applied to all batch elements and heads.
            (batch, 1, 1, length)))


        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = dynamic_vector_slice_in_dim(
            jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

          abs_bias = dynamic_vector_slice_in_dim(
            jnp.squeeze(abs_bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
        mask > 0,
        jnp.full(mask.shape, 0.).astype(self.dtype),
        jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias, abs_bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
      query,
      key,
      value,
      bias=attention_bias,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      deterministic=deterministic,
      dtype=self.dtype,
      float32_logits=self.float32_logits)

    # Back to the original inputs dimensions.
    out = DenseGeneral(
      features=inputs_q.shape[-1],  # output dim is set to the input dim.
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      kernel_axes=('joined_kv', 'embed'),
      dtype=self.dtype,
      name='out')(
      x)
    return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

#------------------------------------------------------------------------------
# Convolution layers
#------------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
  n_e: int
  e_dim: int
  beta: float = 0.25
  embedding_init: Initializer = default_embed_init
  dtype: Any = jnp.float32

  def setup(self):
    self.embedding = param_with_axes(
      'embedding',
      self.embedding_init, (self.n_e, self.e_dim),
      jnp.float32,
      axes=(('vocab', 'embed')))

  def get_codebook_entry(self, indices):
    min_encodings = jax.nn.one_hot(indices, self.n_e, dtype=self.dtype)
    z_q = jnp.einsum('bqk,kd->bqd', min_encodings, self.embedding)
    return z_q

  @nn.compact
  def __call__(self, z: Array) -> Array:

    z_flattened = jnp.reshape(z, (-1, self.e_dim))
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = jnp.sum(z_flattened ** 2, axis=1, keepdims=True) + \
        jnp.sum(self.embedding ** 2, axis=1) - 2 * \
        jnp.einsum('ij,kj->ik', z_flattened, self.embedding)

    min_encoding_indices = jnp.argmin(d, axis=1)
    z_q = jnp.asarray(self.embedding, self.dtype)[min_encoding_indices]
    z_q = jnp.reshape(z_q, z.shape)

    perplexity = None
    min_encodings = None
    loss = jnp.mean((jax.lax.stop_gradient(z_q)-z)**2) + self.beta * \
           jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)

    z_q = z + jax.lax.stop_gradient(z_q - z)

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


def nonlinearity(x):
  # swish
  return x*nn.sigmoid(x)

def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

class Conv(nn.Module):
  """Convolution Module with flexible axes.
    Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Iterable[int]
  strides: Union[None, int, Iterable[int]] = 1
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  input_dilation: Union[None, int, Iterable[int]] = 1
  kernel_dilation: Union[None, int, Iterable[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: DType = jnp.float32
  param_dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  kernel_axes: Tuple[str, ...] = ()
  bias_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.
    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    if isinstance(self.kernel_size, int):
      raise TypeError('The kernel size must be specified as a'
                      ' tuple/list of integers (eg.: [3, 3]).')
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = maybe_broadcast(self.strides)  # self.strides or (1,) * (inputs.ndim - 2)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (
      in_features // self.feature_group_count, self.features)

    kernel = param_with_axes(
      'kernel',
      self.kernel_init,
      kernel_shape,
      self.param_dtype,
      axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    if self.padding == 'CIRCULAR':
      kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
      pads = [(0, 0)] + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    else:
      padding_lax = self.padding

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    y = lax.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision)

    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    if self.use_bias:
      bias = param_with_axes(
        'bias',
        self.bias_init,
        (self.features,),
        self.param_dtype,
        axes=self.bias_axes)

      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y

#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))
    kernel = param_with_axes(
      'kernel',
      self.kernel_init,
      kernel_param_shape,
      jnp.float32,
      axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('linear', 'gelu') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
        self.intermediate_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('embed', 'mlp'),
        name=dense_name)(
        inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
      rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseGeneral(
      inputs.shape[-1],
      dtype=self.dtype,
      kernel_init=self.kernel_init,
      kernel_axes=('mlp', 'embed'),
      name='wo')(
      x)
    return output


class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
      'embedding',
      self.embedding_init, (self.num_embeddings, self.features),
      jnp.float32,
      axes=('vocab', 'embed'))

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  num_buckets: int
  img_num_buckets: int
  max_distance: int
  img_max_distance: int
  num_heads: int
  img_width: int
  img_height: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(jnp.int32) * num_buckets
      n = jnp.abs(n)
    else:
      n = jnp.maximum(n, 0)

    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact + jnp.finfo(jnp.float32).eps) /
        jnp.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(jnp.int32)

    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret += jnp.where(is_small, n, val_if_large)
    return ret

  @staticmethod
  def _img_relative_position_bucket(relative_position_x,
                                    relative_position_y,
                                    num_buckets=8,
                                    max_distance=20):

    max_exact = num_buckets // 2
    nx = -relative_position_x
    ny = -relative_position_y

    total_buckets = num_buckets ** 2
    ret = 0
    ret += (jnp.logical_and(nx <=0, ny <0)).astype(jnp.int32) * total_buckets * 3
    ret += (jnp.logical_and(nx <0, ny >=0)).astype(jnp.int32) * total_buckets * 2
    ret += (jnp.logical_and(nx >0, ny <=0)).astype(jnp.int32) * total_buckets * 1

    nx = jnp.abs(nx)
    ny = jnp.abs(ny)

    is_small_x = nx < max_exact
    val_x_if_large = max_exact + (jnp.log(nx.astype(jnp.float32) /
                                          max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance /
                                                                                            max_exact) * (num_buckets - max_exact)).astype(np.int32)

    val_x_if_large = jnp.minimum(val_x_if_large, num_buckets - 1)

    is_small_y = ny < max_exact
    val_y_if_large = max_exact + (jnp.log(ny.astype(jnp.float32) /
                                          max_exact + jnp.finfo(jnp.float32).eps) / jnp.log(max_distance /
                                                                                            max_exact) * (num_buckets - max_exact)).astype(jnp.int32)
    val_y_if_large = jnp.minimum(val_y_if_large, num_buckets - 1)

    xx = jnp.where(is_small_x, nx, val_x_if_large)
    yy = jnp.where(is_small_y, ny, val_y_if_large)
    ret += xx + num_buckets * yy
    return ret

  @nn.compact
  def __call__(self, txt_position_ids, img_position_ids, bidirectional=True):
    """Produce relative position embedding attention biases.

    Args:
      txt_position_ids: attention query length.
      img_position_ids: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
    # TODO(levskaya): should we be computing this w. numpy as a program
    # constant?

    # compute text position encoding first.
    txt_context_position = txt_position_ids[:, :, None]
    txt_memory_position = txt_position_ids[:, None, :]
    txt_relative_position = txt_memory_position - txt_context_position # shape (qlen, klen)

    # different way to compute relative position.
    rp_bucket = self._relative_position_bucket(
      txt_relative_position,
      bidirectional=bidirectional,
      num_buckets=self.num_buckets,
      max_distance=self.max_distance)

    relative_attention_bias = param_with_axes(
      'rel_embedding',
      self.embedding_init, (self.num_heads, self.num_buckets),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))

    img_position_x = img_position_ids % self.img_width
    img_position_y = img_position_ids // self.img_width
    img_context_position_x = img_position_x[:,:,None]
    img_memory_position_x = img_position_x[:, None, :]
    img_context_position_y = img_position_y[:,:,None]
    img_memory_position_y = img_position_y[:, None, :]
    img_relative_position_x = img_memory_position_x - img_context_position_x
    img_relative_position_y = img_memory_position_y - img_context_position_y

    img_rp_bucket = self._img_relative_position_bucket(
      img_relative_position_x,
      img_relative_position_y,
      num_buckets=self.img_num_buckets,
      max_distance=self.img_max_distance)

    image_num_rel_dis = self.img_num_buckets ** 2 * 4
    img_relative_attention_bias = param_with_axes(
      'image_rel_embedding',
      self.embedding_init, (self.num_heads, image_num_rel_dis),
      jnp.float32,
      axes=('heads', 'relpos_buckets'))

    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    img_relative_attention_bias = jnp.asarray(img_relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)

    img_bcast_iota = lax.broadcasted_iota(jnp.int32, (image_num_rel_dis, 1, 1, 1), 0)
    img_rp_bucket_one_hot = jnp.array(
      img_rp_bucket[jnp.newaxis, ...] == img_bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    t_values = lax.dot_general(
      relative_attention_bias,
      rp_bucket_one_hot,
      (
        ((1,), (0,)),
        ((), ())))  # no batched dims
    i_values = lax.dot_general(
      img_relative_attention_bias,
      img_rp_bucket_one_hot,
      (
        ((1,), (0,)),  # rhs, lhs contracting dims
        ((), ())))  # no batched dims

    t_values_pad = jax.lax.pad(
      t_values,
      jnp.array(0, dtype=t_values.dtype),
      [(0,0,0),(0,0,0),(0,img_position_ids.shape[1],0),(0,img_position_ids.shape[1],0)])

    i_values_pad = jax.lax.pad(
      i_values,
      jnp.array(0, dtype=i_values.dtype),
      [(0,0,0),(0,0,0),(txt_position_ids.shape[1],0,0),(txt_position_ids.shape[1],0,0)])
    values = t_values_pad + i_values_pad
    return jnp.transpose(values, (1,0,2,3))

#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
      'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale

def _canonicalize_axes(rank: int, axes: Axes) -> Iterable[int]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))

def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)

def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics.
  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
      lax.pmean(
        concatenated_mean,
        axis_name=axis_name,
        axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var

def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Any, param_dtype: Any,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Initializer,
               scale_init: Initializer):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
  A seperate bias and scale is learned for each feature as specified by feature_axes.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = param_with_axes('scale', scale_init, reduced_feature_shape,
                            param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = param_with_axes('bias', bias_init, reduced_feature_shape,
                           param_dtype,  axes=('axis_0',)).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)

class GroupNorm(Module):
  num_groups: Optional[int] = 32
  group_size: Optional[int] = None
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = nn.initializers.zeros
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    feature_axes = (-1,)

    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')
    num_groups = self.num_groups

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, self.group_size))
      num_groups = channels // self.group_size

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError('Number of groups ({}) does not divide the number'
                       ' of channels ({}).'.format(num_groups, channels))

    group_size = x.shape[-1] // num_groups
    group_shape = x.shape[:-1] + (num_groups, group_size)

    def broadcast_stat(stat):
      stat = jnp.broadcast_to(stat[..., None], (x.shape[0], num_groups, group_size))
      return stat.reshape((x.shape[0], num_groups * group_size))

    # TODO suport axis_name for model parallelism?
    mean, var = _compute_stats(x.reshape(group_shape), reduction_axes, None, None)
    mean = broadcast_stat(mean)
    var = broadcast_stat(var)

    return _normalize(
      self, x, mean, var, reduction_axes[:-1], feature_axes,
      self.dtype, self.param_dtype, self.epsilon,
      self.use_bias, self.use_scale,
      self.bias_init, self.scale_init)

#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
    # [batch, len_q] -> [batch, len_q, 1]
    jnp.expand_dims(query_input, axis=-1),
    # [batch, len_q] -> [batch, 1, len_kv]
    jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
    idxs,
    idxs,
    jnp.greater_equal,
    extra_batch_dims=extra_batch_dims,
    dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
    f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
    f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, lengtlength]
    inputs_mask = make_attention_mask(
      decoder_causal_attention,
      decoder_causal_attention,
      jnp.logical_and,
      dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
    make_attention_mask(
      decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
      make_attention_mask(
        decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)
