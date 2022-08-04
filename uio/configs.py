from uio.network import UnifiedIOConfig, VAEConfig


DTYPE = "float32"


# Shared between all model sizes
VAE_CONFIG = VAEConfig(
  embed_dim=256,
  n_embed=16384,
  double_z=False,
  z_channels=256,
  resolution=256,
  in_channels=3,
  out_ch=3,
  ch=128,
  ch_mult=(1,1,2,2,4),
  num_res_blocks=2,
  attn_resolutions=(16,),
  dropout=0,
  dtype=DTYPE,
)

CONFIGS = {
  "small": UnifiedIOConfig(
    dtype=DTYPE,
    emb_dim=512,
    num_heads=6,
    num_encoder_layers=8,
    num_decoder_layers=8,
    mlp_dim=1024,
  ),
  "base": UnifiedIOConfig(
    dtype=DTYPE,
    emb_dim=768,
    num_heads=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    mlp_dim=2048,
    vocab_size=33152,
  ),
  "large": UnifiedIOConfig(
    dtype=DTYPE,
    emb_dim=1024,
    num_heads=16,
    num_encoder_layers=24,
    num_decoder_layers=24,
    mlp_dim=2816,
  ),
  "xl": UnifiedIOConfig(
    dtype=DTYPE,
    emb_dim=2048,
    num_heads=32,
    num_encoder_layers=24,
    num_decoder_layers=24,
    mlp_dim=5120,
    num_seg_emb=8
  )
}