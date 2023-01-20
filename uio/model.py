# Modified from code from T5X (https://github.com/google-research/t5x)

import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Type, Union
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions

from uio import decoding

Array = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray]
PyTreeDef = type(jax.tree_util.tree_structure(None))


# Sentinel used instead of None to indicate missing values
_NoValueSentinel = object()


class TokensIdsToLogitsCallable(typing_extensions.Protocol):
  """Token ids to logits mapping call signature."""

  def __call__(
      self, token_ids: jnp.ndarray, cache: Mapping[str, jnp.ndarray]
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Performs forward pass to convert token ids to logits.

    Args:
      token_ids: [batch_size, 1] int32 tokens for single position used during
        incremental decoding. Non-0 prefix tokens to be used as a forced prompt.
      cache: flax attention cache.

    Returns:
      a tuple of logits with a shape [batch_size, vocab_size] and an updated
      cache.
    """
    ...


class DecodeFnCallable(typing_extensions.Protocol):
  """Decoding function call signature."""

  def __call__(self, *, inputs: jnp.ndarray, cache: Mapping[str, jnp.ndarray],
               tokens_to_logits: TokensIdsToLogitsCallable, eos_id: int,
               num_decodes: int, decode_rng: Optional[jnp.ndarray],
               **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Decoding function interface.

    Args:
      inputs: [batch_size, max_decode_len] int32 sequence of tokens, with non-0
        prefix tokens to be used as a forced prompt.
      cache: flax attention cache.
      tokens_to_logits: fast autoregressive decoder function taking single token
        slices and cache and returning next-token logits and updated cache.
      eos_id: end-of-sentence token for target vocabulary.
      num_decodes: number of decoded sequences to be returned.
      decode_rng: an optional JAX PRNG Key for stochastic sampling routines.
      **kwargs: an optional kwargs. One common usecase of this is passing
        decoding parameters at the callsite.

    Returns:
      decodes: Array of sequences: [batch_size, num_decodes, max_decode_len].
        The `num_decodes` dimension is expected to be sorted by the `scores`,
        i.e., `decodes[:, -1, :] has the highest scores among `num_decodes`
        decoded sequences.
      scores: Array of log likelihood scores: [batch_size, num_decodes]
    """
    ...


def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  return loss


class UnifiedIOModel(nn.Module):
  """Wrapper that provides generation methods using a `Transformer` module"""

  def __init__(
      self,
      module: nn.Module,
      text_decoder_length=None,
      image_decoder_length=None,
  ):
    self.module = module
    self._text_decoder_length = text_decoder_length
    self._image_decoder_length = image_decoder_length

  def _compute_logits(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray] = None,
      mutable: flax_scope.CollectionFilter = False
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
    """Computes logits via a forward pass of `self.module_cls`."""
    # Dropout is provided only for the training mode.
    rngs = {'dropout': dropout_rng} if dropout_rng is not None else None

    return self.module.apply(
      {'params': params},
      batch['text_encoder_inputs'],
      batch['image_encoder_inputs'],
      batch['text_decoder_inputs'],
      batch['image_decoder_targets'],
      batch['text_decoder_targets'],
      text_encoder_masks=batch.get('text_encoder_masks', None),
      image_encoder_masks=batch.get('image_input_masks', None),
      image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
      text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
      text_decoder_masks=batch.get('text_decoder_masks', None),
      image_decoder_masks=batch.get('image_target_masks', None),
      text_decoder_segment_ids=batch.get('text_decoder_segment_ids', None),
      text_decoder_positions=batch.get('text_decoder_positions', None),
      cache_text_length=self._text_decoder_length,
      cache_image_length=self._image_decoder_length,
      decode=False,
      enable_dropout=rngs is not None,
      rngs=rngs,
      mutable=mutable)

  def get_initial_variables(
      self,
      rng: jnp.ndarray,
      input_shapes: Mapping[str, Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None,
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    text_encoder_shape = input_shapes['text_encoder_inputs']
    text_encoder_type = input_types.get('text_encoder_inputs', jnp.float32)
    image_encoder_shape = input_shapes['image_encoder_inputs']
    image_encoder_type = input_types.get('image_encoder_inputs', jnp.float32)
    text_decoder_shape = input_shapes['text_decoder_inputs']
    text_decoder_type = input_types.get('text_decoder_inputs', jnp.float32)
    image_decoder_shape = input_shapes['image_decoder_targets']
    image_decoder_type = input_types.get('image_decoder_targets', jnp.float32)
    initial_variables = self.module.init(
      rng,
      jnp.ones(text_encoder_shape, text_encoder_type),
      jnp.ones(image_encoder_shape, image_encoder_type),
      jnp.ones(text_decoder_shape, text_decoder_type),
      jnp.ones(image_decoder_shape, image_decoder_type),
      jnp.ones(text_decoder_shape, text_decoder_type),
      decode=False,
      enable_dropout=False,
      cache_text_length=self._text_decoder_length,
      cache_image_length=self._image_decoder_length,
      vae_decode=True)
    return initial_variables

  def predict_with_answer_options(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      max_options=800,
      average_loss=False
  ):
    text_answer_options = len(batch["output_options"].shape) == 3
    text_encoder_inputs = batch['text_encoder_inputs']
    text_encoder_masks = batch.get('text_encoder_masks')
    if text_encoder_masks is None:
      text_encoder_masks = text_encoder_inputs > 0

    _encoded_inputs, _encoder_masks = self.module.apply(
      {'params': params},
      text_encoder_inputs,
      batch['image_encoder_inputs'],
      text_encoder_masks,
      batch['image_input_masks'],
      image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
      text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
      enable_dropout=False,
      method=self.module.encode
    )

    all_losses = []
    n_options = batch["output_options"].shape[1]

    n_groups = (n_options + max_options - 1) // max_options
    for i in range(n_groups):
      output_options = batch["output_options"][:, i*max_options:(i+1)*max_options]
      batch_size, num_option = output_options.shape[:2]
      encoded, encoder_position_embedding = _encoded_inputs
      encoded = decoding.flat_batch_beam_expand(encoded, num_option)
      encoder_position_embedding = decoding.flat_batch_beam_expand(encoder_position_embedding, num_option)
      encoder_masks = decoding.flat_batch_beam_expand(_encoder_masks, num_option)
      encoded_inputs = (encoded, encoder_position_embedding)
      decoded_size = batch_size*num_option

      if text_answer_options:
        # Text answer options
        # `output_options` does not have EOS or BOS, we need to do a bit work to correctly-formatted
        # text inputs/outputs here
        text_decoder_inputs = output_options.reshape((decoded_size, -1))
        text_decoder_targets = text_decoder_inputs
        text_decoder_targets = jnp.pad(text_decoder_targets, [[0, 0], [0, 1]])  # Add room for EOS

        text_decoder_masks = text_decoder_inputs > 0
        text_decoder_inputs = jnp.pad(text_decoder_inputs, [[0, 0], [1, 0]])
        text_decoder_masks = jnp.pad(text_decoder_masks, [[0, 0], [1, 0]], constant_values=True)

        eos_mask = jnp.logical_and(text_decoder_masks, text_decoder_targets == 0)
        text_decoder_targets = text_decoder_targets + eos_mask

        image_decoder_inputs = jnp.zeros([encoded.shape[0], 1], jnp.int32)
        image_decoder_targets = jnp.zeros([encoded.shape[0], 1], jnp.int32)
        image_decoder_masks = jnp.zeros([encoded.shape[0], 1], jnp.int32)
      else:
        # Image answer options
        image_decoder_masks = batch["output_options_masks"][:, i*max_options:(i+1)*max_options]
        image_decoder_masks = image_decoder_masks.reshape(-1, 256)
        output_options = output_options.reshape([decoded_size] + list(output_options.shape[2:]))

        # Apply the VAE to get the target tokens
        image_decoder_targets = self.module.apply(
          {'params': params},
          output_options,
          method=self.module.encode_target_image
        )

        # Build auto-regressive inputs
        image_start_token = self.module.config.vocab_size - 1
        image_decoder_inputs = jnp.concatenate([
          jnp.zeros((image_decoder_targets.shape[0], 1), dtype=jnp.int32) + image_start_token,
          image_decoder_targets[:, :-1]], axis=1)

        # Predict EOS to start following the training scheme
        text_decoder_inputs = jnp.zeros([decoded_size, 1], jnp.int32)
        text_decoder_targets = jnp.ones([decoded_size, 1], jnp.int32)
        text_decoder_masks = jnp.ones([decoded_size, 1], jnp.int32)

      text_logits, image_logits, image_decoder_targets = self.module.apply(
        {'params': params},
        encoded_inputs,
        encoder_masks,
        text_decoder_inputs,
        image_decoder_inputs,
        text_decoder_targets,
        image_decoder_targets,
        text_decoder_masks=text_decoder_masks,
        image_decoder_masks=image_decoder_masks,
        enable_dropout=False,
        method=self.module.decode
      )

      vocab_size = 33152
      if text_answer_options:
        soft_targets = common_utils.onehot(text_decoder_targets, text_logits.shape[-1], on_value=1.0, off_value=0.0)
        total_loss = cross_entropy_with_logits(text_logits, soft_targets)
        total_loss = total_loss * text_decoder_masks
        total_loss = jnp.sum(total_loss, axis=1)
        if average_loss:
          total_loss = total_loss / jnp.sum(text_decoder_masks, axis=1)
        total_loss = jnp.reshape(total_loss, [batch_size, -1])
      else:
        soft_targets = common_utils.onehot(image_decoder_targets+vocab_size, image_logits.shape[-1])
        total_loss = cross_entropy_with_logits(image_logits, soft_targets)
        total_loss = total_loss * image_decoder_masks
        total_loss = jnp.sum(total_loss, axis=1)
        if average_loss:
          total_loss = total_loss / jnp.sum(image_decoder_masks, axis=1)
        total_loss = jnp.reshape(total_loss, [batch_size, -1])

      all_losses.append(total_loss)

    text_loss = jnp.concatenate(all_losses, -1)
    selected_option_ix = jnp.argmin(text_loss, -1)
    ix = jnp.arange(0, len(selected_option_ix))
    selected_options = batch["output_options"][ix, selected_option_ix]
    selected_loss = text_loss[ix, selected_option_ix]
    out = {'scores': selected_loss, "all_scores": text_loss}
    if text_answer_options:
      out['text_tokens'] = selected_options
    else:
      out['image'] = jnp.clip((selected_options+1)/2.0, 0, 1)
    return out

  def _compute_logits_from_slice(
      self, flat_ids: jnp.ndarray, flat_cache: Mapping[str, jnp.ndarray], cur_index: int,
      live_seqs: jnp.ndarray, params: PyTreeDef, encoded_inputs: jnp.ndarray, encoder_masks: jnp.ndarray,
      text_length: int, image_length: int, logit_masks: jnp.ndarray = None) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]

    def update_flat_ids(x):
      x = jnp.zeros_like(x) + self.module.config.vocab_size - 1
      return x

    def update_pos_ids(x):
      x = x + self.module.config.max_text_length - text_length
      return x

    def identity_fn(x):
      return x

    def update_ones(x):
      x = jnp.zeros_like(x) + 1
      return x

    def update_zeros(x):
      x = jnp.zeros_like(x)
      return x

    flat_ids = jax.lax.cond(
      jax.lax.eq(cur_index, text_length),
      lambda: update_flat_ids(flat_ids),
      lambda: identity_fn(flat_ids))

    seg_ids = jax.lax.cond(
      jax.lax.ge(cur_index, text_length),
      lambda: update_ones(flat_ids),
      lambda: update_zeros(flat_ids))

    decoder_masks = jax.lax.cond(cur_index < text_length,
                                 lambda: jnp.reshape((live_seqs == 1).sum(axis=-1) == 0, (-1,1)),
                                 lambda: jnp.ones(flat_ids.shape, dtype=jnp.bool_))

    flat_logits, new_vars = self.module.apply(
      {
        'params': params,
        'cache': flat_cache
      },
      encoded_inputs,
      encoder_masks,  # only needed for encoder padding mask
      flat_ids,
      decoder_masks=decoder_masks,
      decoder_segments=seg_ids,
      enable_dropout=False,
      decode=True,
      image_decode_length=image_length,
      text_decode_length=text_length,
      cur_index=cur_index,
      mutable=['cache'],
      method=self.module.sample)
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']

    cfg = self.module.config
    total_vocab_size = cfg.vocab_size + cfg.image_vocab_size
    logit_range = jnp.reshape(jnp.arange(total_vocab_size), [1, 1, -1])
    image_logits_mask = jnp.reshape(logit_range < cfg.vocab_size, [1, -1])
    text_logits_mask = jnp.reshape(logit_range >= cfg.vocab_size, [1, -1])

    flat_logits = jax.lax.cond(
      jax.lax.ge(cur_index, text_length),
      lambda: jnp.where(image_logits_mask, -1e10, flat_logits),
      lambda: jnp.where(text_logits_mask, -1e10, flat_logits))

    def update_mask(flat_logits, logit_masks, cur_index):
      mask = jnp.reshape(logit_masks[cur_index], [1, -1])
      flat_logits = jnp.where(mask, -1e10, flat_logits)
      return flat_logits

    # apply mask here.
    if logit_masks is not None:
      flat_logits = jax.lax.cond(
        jax.lax.lt(cur_index, logit_masks.shape[0]),
        lambda: update_mask(flat_logits, logit_masks, cur_index),
        lambda: identity_fn(flat_logits))

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int=1,
      text_length=64,
      image_length=256,
      logit_mask_fn=None,
      beam_search=None,
  ) -> Mapping[str, jnp.ndarray]:
    """Generate outputs from the model.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    if "output_options" in batch:
      return self.predict_with_answer_options(params, batch)

    # [batch, input_len]
    text_encoder_inputs = batch['text_encoder_inputs']
    image_encoder_inputs = batch['image_encoder_inputs']
    image_input_masks = batch['image_input_masks']
    text_encoder_masks = batch.get('text_encoder_masks')
    if text_encoder_masks is None:
      text_encoder_masks = text_encoder_inputs > 0

    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    text_type = batch['text_encoder_inputs'].dtype
    bs = text_encoder_inputs.shape[0]

    _, variables_with_cache = self.module.apply(
      {'params': params},
      jnp.ones_like(text_encoder_inputs),
      jnp.ones_like(image_encoder_inputs),
      jnp.ones((bs, text_length), text_type),
      jnp.ones((bs, 256, 256, 3), image_encoder_inputs.dtype),
      jnp.ones((bs, text_length), text_type),
      decode=True,
      enable_dropout=False,
      vae_decode=False,
      cache_text_length=text_length,
      cache_image_length=image_length,
      mutable=['cache'])

    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded_inputs, encoder_masks = self.module.apply({'params': params},
                                                      text_encoder_inputs,
                                                      image_encoder_inputs,
                                                      text_encoder_masks,
                                                      image_input_masks,
                                                      image_encoder_pos_ids=batch.get('image_encoder_pos_ids', None),
                                                      text_encoder_pos_ids=batch.get('text_encoder_pos_ids', None),
                                                      enable_dropout=False,
                                                      method=self.module.encode)

    encoded, encoder_position_embedding = encoded_inputs
    encoded = decoding.flat_batch_beam_expand(encoded, num_decodes)
    encoder_masks = decoding.flat_batch_beam_expand(encoder_masks, num_decodes)
    encoded_inputs = (encoded, encoder_position_embedding)

    if logit_mask_fn is not None:
      logit_masks = logit_mask_fn()
    else:
      logit_masks = None

    tokens_ids_to_logits = functools.partial(
      self._compute_logits_from_slice,
      params=params,
      encoded_inputs=encoded_inputs,
      encoder_masks=encoder_masks,
      text_length=text_length,
      image_length=image_length,
      logit_masks=logit_masks)

    if decoder_params is None:
      decoder_params = {}

    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decod_prompt_inputs` will be filled with the sampled ids.
    decoder_prompt_inputs = jnp.zeros([bs, text_length+image_length], text_type)

    # TODO(hwchung): rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if isinstance(beam_search, Callable):  # For fine-grain hyper-parameter control
      decodes, scores, logprobs = beam_search(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
      )
    elif beam_search:
      decodes, scores, logprobs = decoding.beam_search(
        inputs=decoder_prompt_inputs,
        cache=cache,
        alpha=0.0,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=1,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        **decoder_params)
    else:
      decodes, scores, logprobs = decoding.temperature_sample(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=1,
        num_decodes=num_decodes,
        topk = 0,
        topp = 0.9,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    scores = jax.lax.stop_gradient(scores)

    out = {}

    if image_length == 256:
      # Get the image tokens and decode with the VAE
      if return_all_decodes:
        image_decodes = decodes[:, :, -256:].reshape(-1, 256)
      else:
        image_decodes = decodes[:, -1, -256:]
      decodes = decodes[:, :, :-256]

      image_decodes = image_decodes - self.module.config.vocab_size
      img = self.module.apply(
        {'params': params},
        method=self.module.decode_code,
        code_b=image_decodes)

      if return_all_decodes:
        img = jnp.reshape(img, decodes.shape[:2] + img.shape[1:])
        image_decodes = jnp.reshape(image_decodes, decodes.shape[:2] + image_decodes.shape[1:])
      out["image"] = jnp.clip((img+1)/2.0, 0, 1)
      out["image_tokens"] = image_decodes

    if not return_all_decodes:
      # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
      # in increasing order of log-probability.
      # Return the highest scoring beam sequence.
      decodes = decodes[:, -1]
      scores = scores[:, -1]

    out["text_tokens"] = decodes
    out["scores"] = scores
    return out
