from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from transformers import T5Tokenizer

from uio.configs import CONFIGS, VAE_CONFIG
from uio import network
from uio import utils
from uio.model import UnifiedIOModel

CAPTIONING_PROMPT = 'What does the image describe ?'
DEPTH_PROMPT = "What is the depth map of the image ?"
SURFACE_NORMAL_PROMPT = 'What is the surface normal of the image ?'
OBJECT_SEGMENTATION = 'What is the segmentation of " {} " ?'
IMAGE_GENERATION = 'What is the complete image? Text: " {} " .'
REFEXP_PROMPT = 'Which region does the text " {} " describe ?'
REGION_CAPTION = 'What does the region " {} " describe ?'
REGION_CLASSIFICATION = 'What is the category of region " {} " ?'
IMAGE_TAGGING = 'What is this in the image ?'
IMAGE_INPAINTING = 'Filling the blank region " {} " ?'
POSE_ESTIMATION = 'Find the human joints in the region " {} " .'
SEGMENTATION_BASED_GENERATION = 'What is the complete image? Segmentation color: " {} "'


GEN_SEGMENTATION_COLORS = np.array([
  [255, 0, 0],
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [255, 255, 255],
  [128, 128, 128],
  [255, 0, 255],
  [255, 255, 0],
  [0, 255, 255],
  [192, 192, 192],
  [128, 0, 0],
  [128, 128, 0],
  [0, 128, 0],
  [0, 128, 128],
  [0, 0, 128],
  [128, 0, 128],
], dtype=np.uint8)


GEN_SEGMENTATION_COLOR_NAMES = [
  "white",
  'red',
  'lime',
  'blue',
  'white',
  'gray',
  'fuchsia',
  'yellow',
  'aqua',
  'silver',
  'maroon',
  'olive',
  'green',
  'teal',
  'navy',
  'purple'
]


class ModelRunner:
  """High-level API to run UnifiedIO

  This is intended to provide an easy way test out examples and
  to demonstrate the pre-/ post-preprocessing we use for different tasks
  """

  def __init__(self, size, param_file, pad_input_to_max=None, max_input_len=64,
               max_options=800, compiled=False, log_inputs=True):
    """Construct the ModeRunner

    :param size: Model size (small, base, large, xl)
    :param param_file: .bin storing the parameters
    :param pad_input_to_max: Always pad input text tokens to this value, this can avoid excess
                       jax.jit re-compilations when `compiled` is set, defaults to the value of `compiled`
    :param max_input_len: if `pad_to_max` is true, the max value to pad to, longer values will
                          result in more expensive inference. We support up 256 token, but
                          we default to 64 which is enough for almost any tasks.
    :param max_options: For input with answer options, max number of options to process at once
    :param compiled: Compile the underlying prediction function, faster inference at a one-time
                     cost when using the same input shapes
    :param log_inputs: Log the input text run on
    """
    self.max_input_len = max_input_len
    if pad_input_to_max is None:
      pad_input_to_max = compiled
    self.pad_to_max = pad_input_to_max
    self.max_options = max_options
    self.compiled = compiled
    self.log_inputs = log_inputs

    conf = CONFIGS[size]
    module = network.Transformer(config=conf, vae_config=VAE_CONFIG)

    logging.info("Setting up model...")
    self.model = UnifiedIOModel(module, text_decoder_length=32, image_decoder_length=1)

    # extra_ids are used as location tokens
    # uio is trained to use at most 256 input tokens
    self.tokenizer = T5Tokenizer.from_pretrained(
      "t5-base", model_max_length=256, extra_ids=1100)

    logging.info("Loading parameters...")
    self.params = utils.load_checkpoint(param_file)
    logging.info("Model is ready")

    self._compiled_batch_fn = None
    self._compiled_option_fn = None

  def _get_batch_fn(self):
    if self.compiled:
      if self._compiled_batch_fn is None:
        self._compiled_batch_fn = jax.jit(
          self.model.predict_batch_with_aux,
          static_argnums=list(range(3, 9)))
      return self._compiled_batch_fn
    else:
      return self.model.predict_batch_with_aux

  def _get_answer_options_fn(self):
    if self.compiled:
      if self._compiled_option_fn is None:
        self._compiled_option_fn = jax.jit(
          self.model.predict_with_answer_options, static_argnums=[2, 3])
      return self._compiled_option_fn
    else:
      return self.model.predict_with_answer_options

  def run(self, input_images, input_texts, output_text_len=128, generate_image=False,
          beam_search=None, num_decodes=None, answer_options=None,
          mask_regions=None, average_loss=False) -> Dict:
    """Runs UnifiedIO on input images/texts and produces output images/text

    :param input_images: List of images as [h, w, 3] float32/uint8 arrays or None
    :param input_texts: List of string prompts
    :param output_text_len: Max text tokens to generate, less max tokens will result in faster
                            inference
    :param generate_image: Generate an image, if false inference will be faster
    :param beam_search: Use beam search rather than sampling, if None using beam_search when
                        not generating an image and sampling otherwise
    :param num_decodes: if `None` return one generation for an input, otherwise generate a list
                        `num_decodes` outputs for each example. Also defines the beam size if
                        doing beam search.
    :param answer_options: List of strings or images, limits text/image generation to one of these options
    :param mask_regions: Mask these regions from ech image, used for inpainting
    :param average_loss: If using answer_options, compute the average per-token loss instead of the
                         total loss
    :return: dictionary outputs with the output text, image, scores and tokens generated
    """
    if answer_options is not None:
      if num_decodes is not None:
        raise NotImplementedError("Not support if `answer_options` is given")

    assert output_text_len <= 128, "128 is the max output text len"
    assert len(input_images) == len(input_texts), "Different number of text/image inputs"

    if beam_search is None:
      beam_search = not generate_image

    input_tokens = np.array(self.tokenizer(
      input_texts, max_length=self.max_input_len, truncation=True,
      padding='max_length' if self.pad_to_max else 'longest')["input_ids"], dtype=np.int32)

    image_tensor = []
    mask_tensor = []
    for ix, image in enumerate(input_images):
      if image is not None:
        assert len(image.shape) == 3 and image.shape[-1] == 3
      image, image_mask = utils.preprocess_image(
        image, None if mask_regions is None else mask_regions[ix])
      image_tensor.append(image)
      mask_tensor.append(image_mask)

    batch = {
      'image_encoder_inputs': np.stack(image_tensor),
      'image_input_masks': np.stack(mask_tensor),
      'text_encoder_inputs': input_tokens,
    }

    if not answer_options:
      if self.log_inputs:
        logging.info(f"Running model text_inputs={input_texts}")
      out = self._get_batch_fn()(
        params=self.params, batch=batch, text_length=output_text_len,
        image_length=256 if generate_image else 1,
        beam_search=beam_search, num_decodes=1 if num_decodes is None else num_decodes,
        return_all_decodes=True
      )
    else:
      if isinstance(answer_options[0], str):
        # One set of strings options for the entire batch
        output_options = np.array(self.tokenizer(
          answer_options, max_length=self.max_input_len, truncation=True,
          padding='longest')["input_ids"], dtype=np.int32)
        output_options = np.expand_dims(output_options, 0)
        bs = len(input_texts)
        output_options = np.tile(output_options, [bs, 1, 1])
        batch["output_options"] = output_options
      elif isinstance(answer_options[0], np.ndarray):
        # One set of image options for the entire batch
        preprocessed = [utils.preprocess_target_image(x) for x in answer_options]
        output_options = np.stack([x[0] for x in preprocessed], 0)
        output_options_mask = np.stack([x[1] for x in preprocessed], 0)
        bs = len(input_texts)
        # [batch, n_options, h, w, c]
        output_options = np.tile(np.expand_dims(output_options, 0), [bs, 1, 1, 1, 1])
        # [batch, n_options, n_patches]
        output_options_mask = np.tile(np.expand_dims(output_options_mask, 0), [bs, 1, 1])
        batch["output_options"] = output_options
        batch["output_options_masks"] = output_options_mask
      else:
        raise NotImplementedError("Per-example answer options")

      if self.log_inputs:
        logging.info(f"Running model text_inputs={input_texts} and "
                     f"{output_options.shape[1]} answer options")
      out = self._get_answer_options_fn()(
        params=self.params, batch=batch, max_options=self.max_options, average_loss=average_loss)
      # Add a fake beam dimensi7on to be compatible with the no answer options case
      out = {k: jnp.expand_dims(v, 1) for k, v in out.items()}

    if generate_image:
      output_image = out["image"]
    else:
      output_image = None

    if output_text_len > 1:
      output_text = []
      for batch_out in out["text_tokens"]:
        beam_text = []
        for beam_out in batch_out:
          row = np.array(beam_out)
          # Manually cutoff at the EOS since jax beam search method will generate tokens beyond it
          eos = np.where(row == 1)[0]
          if len(eos) != 0:
            row = row[:np.min(eos)]
          text = self.tokenizer.decode(row, skip_special_tokens=False)
          beam_text.append(text)
        output_text.append(beam_text)
    else:
      output_text = None

    if num_decodes is None:
      if output_text is not None:
        output_text = [x[0] for x in output_text]
      if output_image is not None:
        output_image = [x[0] for x in output_image]
    outputs = dict(
      text_tokens=np.array(out["text_tokens"]) if "text_tokens" in out else None,
      text=output_text,
      image_tokens=np.array(out["image_tokens"]) if "image_tokens" in out else None,
      image=np.array(output_image),
      score=np.array(out["scores"]),
    )
    if "all_scores" in out:
      outputs["all_scores"] = np.array(out["all_scores"])
    return outputs

  def _extract_text(self, out):
    return {k: out[k][0] for k in ["text", "score"]}

  def _extract_image(self, out):
    return {k: out[k][0] for k in ["image", "score"]}

  def _extract_pose(self, out, image_size):
    tokens = out["text_tokens"][0]
    if len(tokens) == 1:
      points, labels, invalid = utils.extract_keypoints(
        tokens[0], self.tokenizer, image_size)
    else:
      kp = []
      for line in tokens:
        kp.append(utils.extract_keypoints(line, self.tokenizer, image_size))
      points, labels, invalid = utils.transpose_lists(kp)

    out = dict(points=points, labels=labels, invalid=invalid,
               score=out["score"], text_tokens=out["text_tokens"])
    return out

  def _extract_boxes(self, out, image_size, include_labels=False):
    tokens = out["text_tokens"][0]
    if len(tokens) == 1:
      all_labels, all_boxes = utils.tokens_to_regions(tokens[0], image_size)
    else:
      all_boxes = []
      all_labels = []
      for line in tokens:
        labels, boxes = utils.tokens_to_regions(line, image_size)
        all_labels.append(labels)
        all_boxes.append(boxes)

    out = dict(boxes=all_boxes, text=out["text"], score=out["score"], text_tokens=out["text_tokens"])
    if include_labels:
      out["labels"] = all_labels
    return out

  def caption(self, image, num_decodes=None) -> Dict:
    """Generate a caption for `image`"""
    out = self.run([image], [CAPTIONING_PROMPT], output_text_len=32,
                   generate_image=False, num_decodes=num_decodes)
    return self._extract_text(out)

  def vqa(self, image, question, num_decodes=None) -> Dict:
    """Answer `question` for `image`"""
    # We trained on lowercase question so lowercasing is recommended
    out = self.run([image], [question.lower()], output_text_len=32,
                   generate_image=False, num_decodes=num_decodes)
    return self._extract_text(out)

  def depth(self, image, num_decodes=None, beam_search=None) -> Dict:
    """Produce a grayscale depth map for `image`"""
    out = self.run([image], [DEPTH_PROMPT], output_text_len=1, generate_image=True,
                   num_decodes=num_decodes, beam_search=beam_search)
    rescaled_image = utils.undo_image_preprocessing(out["image"][0], image.shape[:2])
    return {
      "image": out["image"][0],
      "rescaled_image": rescaled_image,
      "score": out["score"][0],
    }

  def surface_normal(self, image, num_decodes=None, beam_search=None) -> Dict:
    """Produce a RGB surface normal map for `image`"""
    out = self.run([image], [SURFACE_NORMAL_PROMPT], output_text_len=1, generate_image=True,
                   num_decodes=num_decodes, beam_search=beam_search)
    # Rescale the output image to the size of the original image
    rescaled_image = utils.undo_image_preprocessing(out["image"][0], image.shape[:2])
    return {
      "image": out["image"][0],
      "rescaled_image": rescaled_image,
      "score": out["score"][0],
    }

  def image_generation(self, description, num_decodes=None) -> Dict:
    """Generate an image based on `description`"""
    prompt = IMAGE_GENERATION.replace("{}", description)
    out = self.run(
      [None], [prompt], output_text_len=1, generate_image=True, num_decodes=num_decodes)
    return self._extract_image(out)

  def image_inpainting(self, image, location, replace_with: str, num_decodes=None) -> Dict:
    """Generate an image with `location` in-painted with `replace_with`"""
    region = utils.region_to_tokens(location, image.shape[1], image.shape[0])
    region.append(replace_with)
    prompt = IMAGE_INPAINTING.replace("{}", " ".join(region))
    out = self.run(
      [image], [prompt], output_text_len=1, generate_image=True, num_decodes=num_decodes,
      mask_regions=[np.array(location)]
    )
    return self._extract_image(out)

  def object_segmentation(self, image, object_name, num_decodes=None) -> Dict:
    """Generate instances masks for occurrences of `object_name` in `image`"""
    prompt = OBJECT_SEGMENTATION.replace("{}", object_name)
    out = self.run(
      [image], [prompt], output_text_len=1, generate_image=True, num_decodes=num_decodes)
    if num_decodes is None:
      masks = utils.extract_segmentation_masks(out["image"][0])
    else:
      masks = [utils.extract_segmentation_masks(x) for x in out["image"][0]]
    return dict(mask=masks, image=out["image"][0], score=out["score"][0])

  def refexp(self, image, expression, num_decodes=None) -> Dict:
    """Return the `location` corresponding to `expression`"""
    prompt = REFEXP_PROMPT.replace("{}", expression)
    out = self.run(
      [image], [prompt], output_text_len=32, generate_image=False, num_decodes=num_decodes)
    return self._extract_boxes(out, image.shape)

  def object_localization(self, image, object_name, num_decodes=None) -> Dict:
    """Return the `locations` of `object_name` in `image`"""
    # Same prompt/setup as refex
    return self.refexp(image, object_name, num_decodes)

  def region_caption(self, image, location, num_decodes=None) -> Dict:
    """Generate a caption for `location` in `image`"""
    region = utils.region_to_tokens(location, image.shape[1], image.shape[0])
    prompt = REGION_CAPTION.replace("{}", " ".join(region))
    out = self.run(
      [image], [prompt], output_text_len=32, generate_image=False, num_decodes=num_decodes)
    return self._extract_text(out)

  def region_classification(self, image, location, num_decodes=None, answer_options=None) -> Dict:
    """Return the class of the object in `location` in `image`,
    constrain the outputs to `answer_options` if given"""
    region = utils.region_to_tokens(location, image.shape[1], image.shape[0])
    prompt = REGION_CLASSIFICATION.replace("{}", " ".join(region))
    out = self.run(
      [image], [prompt], output_text_len=32, generate_image=False,
      num_decodes=num_decodes, answer_options=answer_options)
    return self._extract_text(out)

  def image_classification(self, image, num_decodes=None, answer_options=None) -> Dict:
    """Return the class of the `image`, constrain the outputs to `answer_options` if given"""
    out = self.run(
      [image], [IMAGE_TAGGING], output_text_len=32, generate_image=False,
      num_decodes=num_decodes, answer_options=answer_options)
    return self._extract_text(out)

  def pose(self, image, location, num_decodes=None) -> Dict:
    """Return points and labels of human joints in `location`"""
    region = utils.region_to_tokens(location, image.shape[1], image.shape[0])
    prompt = POSE_ESTIMATION.replace("{}", " ".join(region))
    out = self.run(
      [image], [prompt], output_text_len=128, generate_image=False,
      num_decodes=num_decodes, beam_search=False)
    return self._extract_pose(out, image.shape[:2])

  def segmentation_based_generation(
      self, binary_masks: List[np.ndarray], labels: List[str], num_decodes=None) -> Dict:
    """Return an image where pixels in each `binary_mask` belong to corresponding class in
      `labels"""
    assert len(binary_masks) <= len(GEN_SEGMENTATION_COLOR_NAMES)
    assert len(binary_masks) == len(labels)
    assert len(binary_masks) > 0

    h, w = binary_masks[0].shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for ix, mask in enumerate(binary_masks):
      image[mask, :] = GEN_SEGMENTATION_COLORS[ix]
    text = " , ".join(f"{a} : {b}" for a, b in zip(labels, GEN_SEGMENTATION_COLOR_NAMES))
    text = text.lower()
    prompt = SEGMENTATION_BASED_GENERATION.replace("{}", text)
    out = self.run(
      [image], [prompt], output_text_len=1, generate_image=True, num_decodes=num_decodes)
    return self._extract_image(out)
