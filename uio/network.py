"""Defines the modules that make up the UnifiedIO model"""
# Modified from code from T5X (https://github.com/google-research/t5x)

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import math

import jax
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import uio.t5x_layers as layers


@dataclass
class UnifiedIOConfig:
  vocab_size: int = 33152
  image_vocab_size: int = 16384
  image_patch_size: int = 16
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  dropout_rate: float = 0.0
  # the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = True
  # Whether to accumulate attention logits in float32 regardless of dtype.
  float32_attention_logits: bool = False
  encoder_max_image_length: int = 576
  encoder_max_text_length: int = 256
  decoder_max_image_length: int = 256
  decoder_max_text_length: int = 256
  visual_backbone_type: str = None
  visual_backbone_feature: str = None
  default_image_size: Sequence[int] = (384, 384)
  num_seg_emb: int = 2


@dataclass
class VAEConfig:
  embed_dim: int = 256
  n_embed: int = 1024
  double_z: bool = False
  z_channels: int = 256
  resolution: int = 256
  in_channels: int = 3
  out_ch: int = 3
  ch: int = 128
  ch_mult: Sequence[int] = (1,1,2,2,4)
  num_res_blocks: int = 2
  attn_resolutions: Sequence[int] = (16,)
  dropout: float = 0
  dtype: Any = jnp.float32


class AttnBlock(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    h_ = x
    h_ = layers.GroupNorm(name='norm')(h_)
    q = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='q')(h_)

    k = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='k')(h_)

    v = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='v')(h_)

    b, h, w, c = q.shape

    w_ = jnp.einsum('bqc,bkc->bqk', jnp.reshape(q, (b, h*w, c)), jnp.reshape(k, (b, h*w, c)))
    w_ = w_ * (c ** -0.5)
    w_ = jax.nn.softmax(w_).astype(self.dtype)
    h_ = jnp.einsum('bqk,bkd->bqd', w_, jnp.reshape(v, (b, h*w, c)))
    h_ = jnp.reshape(h_, (b, h, w, c))
    h_ = layers.Conv(
      features=self.n_in,
      kernel_size=(1, 1),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='proj_out')(h_)

    return x+h_


class Downsample(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    return layers.Conv(
      features=self.n_in,
      kernel_size=(3, 3),
      strides=(2,2),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv')(x)


class Upsample(nn.Module):
  n_in: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    B, H, W, C = x.shape
    x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')
    x = layers.Conv(
      features=self.n_in,
      kernel_size=(3, 3),
      strides=(1,1),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv')(x)

    return x


class ResBlock(nn.Module):
  n_in: int
  n_out: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, training=False):
    h = x
    h = layers.GroupNorm(name='norm1')(h)
    h = layers.nonlinearity(h)
    h = layers.Conv(
      features=self.n_out,
      kernel_size=(3, 3),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv1')(h)

    h = layers.GroupNorm(name='norm2')(h)
    h = layers.nonlinearity(h)
    h = layers.Conv(
      features=self.n_out,
      kernel_size=(3, 3),
      dtype=self.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv2')(h)

    if self.n_in != self.n_out:
      x = layers.Conv(
        features=self.n_out,
        kernel_size=(1,1),
        dtype=self.dtype,
        kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
        bias_axes=('axis_3',),
        name='nin_shortcut')(x)
    return x + h
  
  
class VAE_Encoder(nn.Module):
  """Jax implementation of Taming VAE encoder"""
  config: VAEConfig

  @nn.compact
  def __call__(self, x, training=False):
    cfg = self.config
    curr_res = cfg.resolution
    num_resolutions = len(cfg.ch_mult)
    in_ch_mult = (1,)+tuple(cfg.ch_mult)

    hs = layers.Conv(
      features=1 * cfg.ch,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_in')(x)

    for i_level in range(num_resolutions):
      block_in = cfg.ch * in_ch_mult[i_level]
      block_out = cfg.ch * cfg.ch_mult[i_level]
      for i_block in range(cfg.num_res_blocks):
        hs = ResBlock(
          block_in,
          block_out,
          cfg.dtype,
          name=f"down_{i_level}_block_{i_block}")(hs)
        block_in = block_out
        if curr_res in cfg.attn_resolutions:
          hs = AttnBlock(
            block_in,
            name=f"down_{i_level}_attn_{i_block}")(hs)

      if i_level != num_resolutions-1:
        hs = Downsample(
          block_in,
          name=f"down_{i_level}_downsample")(hs)
        curr_res = curr_res // 2

    hs = ResBlock(block_in, block_in, name='mid_block_1')(hs)
    hs = AttnBlock(block_in, name='mid_attn_1')(hs)
    hs = ResBlock(block_in, block_in, name='mid_block_2')(hs)
    hs = layers.GroupNorm(name='norm_out')(hs)

    hs = layers.nonlinearity(hs)
    hs = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(3, 3),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_out')(hs)

    return hs

class VAE_Decoder(nn.Module):
  """Jax implementation of Taming VAE encoder"""
  config: VAEConfig

  @nn.compact
  def __call__(self, x, training=False):

    cfg = self.config
    in_ch_mult = (1,)+tuple(cfg.ch_mult)
    num_resolutions = len(cfg.ch_mult)
    curr_res = cfg.resolution // 2**(num_resolutions-1)
    block_in = cfg.ch*cfg.ch_mult[num_resolutions-1]

    # z to block_in
    h = layers.Conv(
      features=block_in,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_in')(x)

    h = ResBlock(block_in, block_in, name='mid_block_1')(h)
    h = AttnBlock(block_in, name='mid_attn_1')(h)
    h = ResBlock(block_in, block_in, name='mid_block_2')(h)

    for i_level in reversed(range(num_resolutions)):
      i_idx = num_resolutions - i_level-1
      block_out = cfg.ch * cfg.ch_mult[i_level]
      for i_block in range(cfg.num_res_blocks+1):
        h = ResBlock(block_in, block_out, name=f"up_{i_idx}_block_{i_block}")(h)
        block_in = block_out
        if curr_res in cfg.attn_resolutions:
          h = AttnBlock(block_in, name=f"up_{i_idx}_attn_{i_block}")(h)
      if i_level != 0:
        h = Upsample(block_in, name=f"up_{i_idx}_upsample")(h)
        curr_res = curr_res * 2

    h = layers.GroupNorm(name='norm_out')(h)
    h = layers.nonlinearity(h)
    h = layers.Conv(
      features=cfg.out_ch,
      kernel_size=(3, 3),
      strides=(1, 1),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='conv_out')(h)

    return h

class DiscreteVAE(nn.Module):
  """Jax implementation of Taming VAE"""
  config: VAEConfig

  def setup(self):
    cfg = self.config
    self.encoder = VAE_Encoder(cfg)
    self.quant_conv = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(1, 1),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='quant_conv')

    self.quantize = layers.VectorQuantizer(
      cfg.n_embed,
      cfg.embed_dim,
      beta=0.25)

    self.post_quant_conv = layers.Conv(
      features=cfg.z_channels,
      kernel_size=(1, 1),
      dtype=cfg.dtype,
      kernel_axes=('axis_0', 'axis_1', 'axis_2', 'axis_3'),
      bias_axes=('axis_3',),
      name='post_quant_conv')

    self.decoder = VAE_Decoder(cfg)

  def encode(self, x, training=False):
    h = self.encoder(x, training)
    h = self.quant_conv(h)
    quant, emb_loss, info = self.quantize(h)
    return quant, emb_loss, info

  def decode(self, quant, training=False):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant, training)
    return dec

  def decode_code(self, code_b):
    quant_b = self.quantize.get_codebook_entry(code_b)
    bs, seq_len, dim = quant_b.shape
    size = int(math.sqrt(seq_len))
    quant_b = jnp.reshape(quant_b, (bs, size, size, dim))
    dec = self.decode(quant_b)
    return dec

  def get_codebook_indices(self, x, vae_decode=False, training=False):
    h = self.encoder(x, training)
    h = self.quant_conv(h)
    z, _, [_, _, indices] = self.quantize(h)

    if vae_decode:
      _ = self.decode(z, training)

    return jnp.reshape(indices, (jnp.shape(h)[0], -1))

  @nn.compact
  def __call__(self, x, training=False):
    quant, diff, _ = self.encode(x, training)
    dec = self.decode(quant, training)
    return dec

class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: UnifiedIOConfig
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self, inputs, txt_position_ids, img_position_ids, abs_pos_bias, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Relative position embedding as attention biases.
    encoder_bias = self.relative_embedding(txt_position_ids, img_position_ids,
                                           True)
    # Attention block.
    assert inputs.ndim == 3
    x = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_attention_layer_norm')(
      inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='attention')(
      x, x, encoder_mask, encoder_bias, abs_pos_bias, deterministic=deterministic)

    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)

    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
      intermediate_dim=cfg.mlp_dim,
      activations=cfg.mlp_activations,
      intermediate_dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      name='mlp',
    )(y, deterministic=deterministic)

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)
    y = y + x
    return y

class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: UnifiedIOConfig
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               self_abs_pos_bias,
               cross_abs_pos_bias,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               image_decoder_positions=None,
               text_decoder_positions=None):

    cfg = self.config

    # Relative position embedding as attention biases.
    # l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relative_embedding(text_decoder_positions, image_decoder_positions, False)
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
      inputs)
    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='self_attention')(
      x,
      x,
      decoder_mask,
      decoder_bias,
      self_abs_pos_bias,
      deterministic=deterministic,
      decode=decode)

    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)

    x = x + inputs
    # Encoder-Decoder block.
    y = layers.LayerNorm(
      dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
      x)
    y = layers.MultiHeadDotProductAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      dropout_rate=cfg.dropout_rate,
      float32_logits=cfg.float32_attention_logits,
      name='encoder_decoder_attention')(
      y,
      encoded,
      encoder_decoder_mask,
      None,
      cross_abs_pos_bias,
      deterministic=deterministic)


    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)

    y = y + x

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = layers.MlpBlock(
      intermediate_dim=cfg.mlp_dim,
      activations=cfg.mlp_activations,
      intermediate_dropout_rate=cfg.dropout_rate,
      dtype=cfg.dtype,
      name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      z, deterministic=deterministic)
    z = z + y

    return z


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: UnifiedIOConfig
  shared_embedding: nn.Module

  def setup(self):
    cfg = self.config
    self.segment_embedding = layers.Embed(
      num_embeddings=cfg.num_seg_emb,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='segment_embedding')

    self.positon_embedding = layers.Embed(
      num_embeddings=cfg.encoder_max_text_length+cfg.encoder_max_image_length,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='position_embedding')

  @nn.compact
  def __call__(self,
               text_encoder_inputs,
               image_encoder_inputs,
               txt_position_ids,
               img_position_ids,
               encoder_masks=None,
               deterministic=False):
    cfg = self.config
    assert text_encoder_inputs.ndim == 2  # [batch, length]
    if image_encoder_inputs.ndim == 3:
      # use default length
      bs = image_encoder_inputs.shape[0]
      h, w = cfg.default_image_size
    else:
      bs, h, w, _ = image_encoder_inputs.shape

    txt_length = text_encoder_inputs.shape[1]

    rel_emb = layers.RelativePositionBiases(
      num_buckets=32,
      img_num_buckets=8,
      max_distance=128,
      img_max_distance=20,
      num_heads=cfg.num_heads,
      img_width=w//cfg.image_patch_size,
      img_height=h//cfg.image_patch_size,
      dtype=cfg.dtype,
      embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                      'uniform'),
      name='relpos_bias')

    # do the image encoding.
    if image_encoder_inputs.ndim == 4:
      img_emb = layers.space_to_depth(image_encoder_inputs, spatial_block_size=cfg.image_patch_size)
    else:
      img_emb = image_encoder_inputs

    txt_pos_emb = self.positon_embedding(txt_position_ids)
    img_pos_emb = self.positon_embedding(img_position_ids+cfg.encoder_max_text_length)

    if (image_encoder_inputs.ndim == 4 and
        img_emb.shape[1] != cfg.encoder_max_image_length and
        img_emb.shape[1] != 1):
      # Our input is a full-sized image that has more or less patches than our default
      # `img_emb.shape[1] != 1` catches the case of being give
      pos_size = int(cfg.encoder_max_image_length ** 0.5)
      target_size = int(img_emb.shape[1] ** 0.5)
      img_pos_emb = jnp.reshape(img_pos_emb, [1, pos_size, pos_size, cfg.emb_dim])
      img_pos_emb = jax.image.resize(img_pos_emb, [1, target_size, target_size, cfg.emb_dim], "bicubic")
      img_pos_emb = jnp.reshape(img_pos_emb, [1, -1, cfg.emb_dim])
      # update image position ids for relative position encoding.
      img_position_ids = jnp.arange(img_emb.shape[1], dtype=jnp.int32)
      img_position_ids = jnp.expand_dims(img_position_ids, axis=0)

    img_emb = layers.DenseGeneral(
      cfg.emb_dim,
      dtype=cfg.dtype,
      kernel_axes=('image_patch', 'embed'),
      name='image_projection',
    )(img_emb)

    # do the text encoding
    # [batch, length] -> [batch, length, emb_dim]
    txt_emb = self.shared_embedding(text_encoder_inputs.astype('int32'))

    txt_segments = jnp.zeros(txt_emb.shape[1], dtype=jnp.int32)[None,...]
    img_segments = jnp.ones(img_emb.shape[1],  dtype=jnp.int32)[None,...]

    txt_emb += self.segment_embedding(txt_segments)
    img_emb += self.segment_embedding(img_segments)

    txt_emb += txt_pos_emb
    img_emb += img_pos_emb

    txt_emb = layers.LayerNorm(
      dtype=cfg.dtype, name='txt_emb_pre_ln')(txt_emb)

    img_emb = layers.LayerNorm(
      dtype=cfg.dtype, name='img_emb_pre_ln')(img_emb)

    position_embedding =jnp.concatenate([txt_pos_emb, img_pos_emb], axis=1)

    position_embedding = layers.LayerNorm(
      dtype=cfg.dtype, name='pe_pre_ln')(position_embedding)

    # get absolute position bias.
    pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_q_linear',
    )(position_embedding)

    pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='position_k_linear',
    )(position_embedding)

    pos_scaling = float(cfg.emb_dim / cfg.num_heads) ** -0.5
    abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', pos_q, pos_k) * pos_scaling

    x = jnp.concatenate([txt_emb, img_emb], axis=1)
    x = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
        config=cfg, relative_embedding=rel_emb,
        name=f'layers_{lyr}')(x, txt_position_ids, img_position_ids, abs_pos_bias, encoder_masks, deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic), position_embedding


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: UnifiedIOConfig
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoded,
               decoder_inputs,
               decoder_positions=None,
               decoder_segments=None,
               decoder_attn_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               image_decoder_positions=None,
               text_decoder_positions=None,
               cur_index=None):

    cfg = self.config
    assert decoder_inputs.ndim == 2  # [batch, len]
    encoded, encoder_position_embedding = encoded

    rel_emb = layers.RelativePositionBiases(
      num_buckets=32,
      img_num_buckets=8,
      max_distance=128,
      img_max_distance=20,
      num_heads=cfg.num_heads,
      img_width=16,
      img_height=16,
      dtype=cfg.dtype,
      embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                      'uniform'),
      name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_inputs.astype('int32'))

    position_embedding = layers.Embed(
      num_embeddings=cfg.decoder_max_text_length + cfg.decoder_max_image_length,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='position_embedding')(decoder_positions)

    if cur_index is None:
      y += position_embedding
    else:
      y += position_embedding[:,cur_index][:,None,:]

    y += layers.Embed(
      num_embeddings=cfg.num_seg_emb,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='segments_embedding')(decoder_segments)

    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_ln')(y)

    position_embedding = layers.LayerNorm(
      dtype=cfg.dtype, name='pe_pre_ln')(position_embedding)

    # get absolute position bias.
    self_pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='self_position_q_linear',
    )(position_embedding)

    self_pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='self_position_k_linear',
    )(position_embedding)

    pos_scaling = float(cfg.emb_dim / cfg.num_heads) ** -0.5
    self_abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', self_pos_q, self_pos_k) * pos_scaling

    # get absolute position bias.
    cross_pos_q = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='cross_position_q_linear',
    )(position_embedding)

    cross_pos_k = layers.DenseGeneral(
      features=(cfg.num_heads, cfg.head_dim),
      dtype=cfg.dtype,
      kernel_axes=('embed', 'joined_kv'),
      name='cross_position_k_linear',
    )(encoder_position_embedding)

    cross_abs_pos_bias = jnp.einsum('bqhd,bkhd->bhqk', cross_pos_q, cross_pos_k) * pos_scaling

    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = DecoderLayer(
        config=cfg,
        relative_embedding=rel_emb,
        name=f'layers_{lyr}')(
        y,
        encoded,
        self_abs_pos_bias,
        cross_abs_pos_bias,
        decoder_mask=decoder_attn_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=deterministic,
        decode=decode,
        image_decoder_positions=image_decoder_positions,
        text_decoder_positions=text_decoder_positions)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
      rate=cfg.dropout_rate, broadcast_dims=(-2,))(
      y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = layers.DenseGeneral(
        cfg.vocab_size + cfg.image_vocab_size,
        dtype=jnp.float32,  # Use float32 for stabiliity.
        kernel_axes=('embed', 'vocab'),
        name='logits_dense')(y)

    return logits


class Transformer(nn.Module):
  """The ynderlying UnifiedIO network"""

  config: UnifiedIOConfig
  vae_config: VAEConfig

  def setup(self):
    cfg = self.config
    vae_config = self.vae_config

    self.shared_embedding = layers.Embed(
      num_embeddings=cfg.vocab_size + cfg.image_vocab_size,
      features=cfg.emb_dim,
      dtype=cfg.dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='token_embedder')

    self.discrete_vae = DiscreteVAE(config=vae_config)
    self.encoder = Encoder(
      config=cfg,
      shared_embedding=self.shared_embedding,
    )
    self.decoder = Decoder(
      config=cfg,
      shared_embedding=self.shared_embedding)

    total_vocab_size = cfg.vocab_size + cfg.image_vocab_size
    self.logit_range = jnp.reshape(jnp.arange(total_vocab_size), [1, 1, -1])
    self.image_logits_mask = jnp.reshape(self.logit_range < cfg.vocab_size, [1, -1])
    self.text_logits_mask = jnp.reshape(self.logit_range >= cfg.vocab_size, [1, -1])

  def encode(self,
             text_encoder_inputs,
             image_encoder_inputs,
             text_encoder_masks,
             image_encoder_masks,
             image_encoder_pos_ids,
             text_encoder_pos_ids,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert text_encoder_inputs.ndim == 2  # (batch, len)
    bs = text_encoder_inputs.shape[0]

    if text_encoder_masks is None:
      text_encoder_masks = text_encoder_inputs > 0

    if image_encoder_inputs.ndim == 3:
      image_length = image_encoder_inputs.shape[1]
    else:
      image_length = int(np.prod(image_encoder_inputs.shape[1:3]) / (cfg.image_patch_size**2))

    if image_encoder_masks is None:
      image_encoder_masks = jnp.ones([bs, image_length], dtype=jnp.bool_)

    if image_encoder_pos_ids is None:
      image_encoder_pos_ids = jnp.arange(image_length, dtype=jnp.int32)
      image_encoder_pos_ids = jnp.expand_dims(image_encoder_pos_ids, axis=0)
      image_encoder_pos_ids = jnp.tile(image_encoder_pos_ids, [bs, 1])

    if text_encoder_pos_ids is None:
      text_encoder_pos_ids = jnp.arange(text_encoder_inputs.shape[1], dtype=jnp.int32)
      text_encoder_pos_ids = jnp.expand_dims(text_encoder_pos_ids, axis=0)
      text_encoder_pos_ids = jnp.tile(text_encoder_pos_ids, [bs, 1])

    encoder_masks = jnp.concatenate([text_encoder_masks, image_encoder_masks], axis=1)
    encoder_attn_masks = layers.make_attention_mask(
      encoder_masks, encoder_masks, dtype=cfg.dtype)

    return self.encoder(
      text_encoder_inputs,
      image_encoder_inputs,
      text_encoder_pos_ids,
      image_encoder_pos_ids,
      encoder_attn_masks,
      deterministic=not enable_dropout
    ), encoder_masks

  def decode(
      self,
      encoded,
      encoder_masks,
      text_decoder_inputs,
      image_decoder_inputs,
      text_decoder_targets,
      image_decoder_targets,
      text_decoder_masks=None,
      image_decoder_masks=None,
      text_decoder_segment_ids=None,
      text_decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    if text_decoder_masks is None:
      text_decoder_masks = text_decoder_targets > 0

    if image_decoder_masks is None:
      image_decoder_masks = jnp.ones(image_decoder_inputs.shape, dtype=jnp.bool_)

    if text_decoder_segment_ids is not None:
      decoder_segment_ids = jnp.concatenate([text_decoder_segment_ids, jnp.ones(image_decoder_masks.shape)], axis=1)
    else:
      decoder_segment_ids = None

    decoder_masks = jnp.concatenate([text_decoder_masks, image_decoder_masks], axis=1)
    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=decoder_masks,
      dtype=cfg.dtype,
      decoder_segment_ids=decoder_segment_ids)

    encoder_decoder_mask = layers.make_attention_mask(
      decoder_masks, encoder_masks, dtype=cfg.dtype)

    decoder_inputs = jnp.concatenate([text_decoder_inputs, image_decoder_inputs], axis=1)

    if text_decoder_positions is None:
      text_decoder_positions = jnp.arange(text_decoder_inputs.shape[1], dtype=jnp.int32)[None,...]
      image_decoder_positions = jnp.arange(image_decoder_inputs.shape[1], dtype=jnp.int32)[None,...]
    else:
      image_decoder_positions = jnp.arange(image_decoder_inputs.shape[1], dtype=jnp.int32)[None,...]
      image_decoder_positions = jnp.tile(image_decoder_positions, [image_decoder_inputs.shape[0], 1])

    decoder_positions = jnp.concatenate([
      text_decoder_positions,
      cfg.decoder_max_text_length+image_decoder_positions],
      axis=1)

    decoder_segments = jnp.expand_dims(
      jnp.concatenate([
        jnp.zeros(text_decoder_inputs.shape[1], dtype=jnp.int32),
        jnp.ones(image_decoder_inputs.shape[1], dtype=jnp.int32)],
        axis=0),
      axis=0)

    logging.info(f"Decode called with EncodeLen={encoded[0].shape[1]}, DecodeInputLen={decoder_inputs.shape[1]}")
    logits = self.decoder(
      encoded,
      decoder_positions=decoder_positions,
      decoder_segments=decoder_segments,
      decoder_inputs=decoder_inputs,
      decoder_attn_mask=decoder_attn_mask,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      image_decoder_positions=image_decoder_positions,
      text_decoder_positions=text_decoder_positions)

    # mask the logits.
    text_length = text_decoder_inputs.shape[1]
    seq_range = jnp.reshape(jnp.arange(logits.shape[1]), [1, -1, 1])
    logits_mask = (((seq_range >= text_length) & (self.logit_range < cfg.vocab_size)) |
                   (seq_range < text_length) & (self.logit_range >= cfg.vocab_size))
    logits = jnp.where(logits_mask, -1e10, logits)
    text_logits = logits[:,:text_length]
    image_logits = logits[:,text_length:]

    return text_logits, image_logits, image_decoder_targets

  def decode_code(self, code_b):
    return self.discrete_vae.decode_code(code_b)

  def encode_target_image(self, image):
    return self.discrete_vae.get_codebook_indices(image)

  def sample(
      self,
      encoded,
      encoder_masks,
      decoder_inputs,
      decoder_masks=None,
      decoder_segments=None,
      enable_dropout=True,
      decode=False,
      cur_index=None,
      image_decode_length=None,
      text_decode_length=None):

    cfg = self.config
    encoder_decoder_mask = layers.make_attention_mask(
      jnp.ones_like(decoder_inputs),
      encoder_masks,
      dtype=cfg.dtype)

    if decoder_masks is not None:
      decoder_attn_mask = layers.make_decoder_mask(
        decoder_target_tokens=decoder_masks,
        dtype=cfg.dtype)
    else:
      decoder_attn_mask = None

    image_decoder_positions = jnp.arange(image_decode_length)[None,...]
    text_decoder_positions = jnp.arange(text_decode_length)[None,...]

    decoder_positions = jnp.concatenate([
      text_decoder_positions,
      cfg.decoder_max_text_length+image_decoder_positions],
      axis=1)

    logits = self.decoder(
      encoded,
      decoder_inputs=decoder_inputs,
      decoder_positions=decoder_positions,
      decoder_segments=decoder_segments,
      decoder_attn_mask=decoder_attn_mask,
      encoder_decoder_mask=encoder_decoder_mask,
      deterministic=not enable_dropout,
      decode=decode,
      image_decoder_positions=image_decoder_positions,
      text_decoder_positions=text_decoder_positions,
      cur_index=cur_index)

    return logits

  def __call__(self,
               text_encoder_inputs,
               image_encoder_inputs,
               text_decoder_inputs,
               image_decoder_targets,
               text_decoder_targets,
               text_encoder_masks=None,
               image_encoder_masks=None,
               text_decoder_masks=None,
               image_decoder_masks=None,
               image_encoder_pos_ids=None,
               text_encoder_pos_ids=None,
               text_decoder_segment_ids=None,
               text_decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               cache_text_length = None,
               cache_image_length = None,
               vae_decode: bool = False,
               return_targets = False
               ):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    cfg = self.config

    if image_decoder_targets.shape[1] > 1:
      image_decoder_tokens = self.discrete_vae.get_codebook_indices(image_decoder_targets, vae_decode) # 0 is the start token.
      # stop gradient.
      image_decoder_tokens = image_decoder_tokens + cfg.vocab_size
      image_decoder_tokens = jax.lax.stop_gradient(image_decoder_tokens)
    else:
      # Dummy input image, use a single token as output that will be masked out
      bs = image_decoder_targets.shape[0]
      image_decoder_tokens = jnp.zeros((bs, 1), dtype=jnp.int32)
      # Client should ensure this are also size 1 and mask out the image token
      assert image_decoder_targets.shape[1] == 1
      if image_decoder_masks is not None:
        assert image_decoder_masks.shape[1] == 1

    image_decoder_inputs = jnp.concatenate([
      jnp.zeros((image_decoder_tokens.shape[0], 1), dtype=jnp.int32) + cfg.vocab_size - 1,
      image_decoder_tokens[:,:-1]], axis=1)

    encoded, encoder_masks = self.encode(
      text_encoder_inputs,
      image_encoder_inputs,
      text_encoder_masks,
      image_encoder_masks,
      image_encoder_pos_ids,
      text_encoder_pos_ids,
      enable_dropout=enable_dropout)

    if cache_image_length is not None:
      image_decoder_inputs = image_decoder_inputs[:,:cache_image_length]
      image_decoder_tokens = image_decoder_tokens[:,:cache_image_length]
      if image_decoder_masks is not None:
        image_decoder_masks = image_decoder_masks[:,:cache_image_length]

    if cache_text_length is not None:
      text_decoder_inputs = text_decoder_inputs[:,:cache_text_length]
      text_decoder_targets = text_decoder_targets[:,:cache_text_length]
      if text_decoder_masks is not None:
        text_decoder_masks = text_decoder_masks[:,:cache_text_length]

    logits = self.decode(
      encoded,
      encoder_masks,
      text_decoder_inputs,
      image_decoder_inputs,
      text_decoder_targets,
      image_decoder_tokens,
      text_decoder_masks=text_decoder_masks,
      image_decoder_masks=image_decoder_masks,
      text_decoder_segment_ids=text_decoder_segment_ids,
      text_decoder_positions=text_decoder_positions,
      enable_dropout=enable_dropout,
      decode=decode)

    if return_targets:
      return logits
    else:
      return logits


