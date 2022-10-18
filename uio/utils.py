from typing import List, Tuple, Union, Sequence

import jax
import numpy as np
import jax.numpy as jnp
from flax.serialization import from_bytes

# Constants used by all UnifiedIO Models
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

vocab_size = 33100
BIN_START = vocab_size - 1100
NUM_DETECTION_BIN = 1000
VOCAB_START = 100
IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_PATCH_SIZE = 16

IMAGE_TARGET_SIZE = [256, 256]


def load_checkpoint(checkpoint):
  """Load a bin file as a tree of jax arrays"""
  with open(checkpoint, "rb") as state_f:
    state = from_bytes(None, state_f.read())
  state = jax.tree_util.tree_map(jnp.array, state)
  return state


def transpose_lists(lsts):
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def region_to_tokens(box, img_w, img_h) -> List[str]:
  """Convert a region into a text sequence

  :param box: [x1, y1, x2, y2] non-normalized bounding box
  :param img_w: image width
  :param img_h: image height
  :return: text tokens representation of the region
  """
  # Convert to yx format normalized to the padded input image.
  scale = max(img_w, img_h)
  box = np.array([box[1], box[0], box[3], box[2]]) / scale
  # Quantize
  quantized_boxes = ((NUM_DETECTION_BIN-1) * box).astype(np.int32)
  # Convert to tokens
  return [f"<extra_id_{i+100}>" for i in quantized_boxes]


def tokens_to_regions(predicted_tokens, image_size, token_per_label=4) -> Tuple[List[str], np.ndarray]:
  """Convert tokens into a list of image locations and labels

  :param predicted_tokens: Integer tokens from UnifiedIO
  :param image_size: original image size
  :param token_per_label: number of location tokens preceding each object label
  :return:
    labels: List[str] of object labels
    locations: np.ndarray [n_objects, token_per_label] image coordinates
  """
  predicted_tokens = np.array(predicted_tokens)
  locations = []
  labels = []
  cur = 0
  while True:
    if cur >= len(predicted_tokens) or predicted_tokens[cur] == 1:
      # end of sequence
      break
    if not np.all(predicted_tokens[cur:cur+token_per_label] > BIN_START):
      # error, should be a list of locations then label
      raise ValueError()
    locations.append(vocab_size-predicted_tokens[cur:cur+token_per_label] - 100)
    cur += token_per_label
    label_end = cur
    while label_end < len(predicted_tokens) and 1 < predicted_tokens[label_end] <= BIN_START:
      label_end += 1
    labels.append(predicted_tokens[cur:label_end])
    cur = label_end

  locations = np.array(locations)
  locations = locations.reshape((-1, 2))[:, ::-1].reshape((-1, token_per_label))  # [yx to xy]
  # Account for image resizing
  factor = max(image_size)
  locations = locations * (factor / 1000)
  return labels, locations


def extract_keypoints(tokens, tokenizer, image_size):
  """Read keypoints from UnifiedIO output

  :param tokens: integer tokens generated
  :param tokenizer: T5Tokenizer
  :param image_size: size of the input image
  :return:
    points: [17, 2] keypoint coordinates
    labels: [17] integer labels between 0 and 2
    invalid: bool, true if `tokens` did not correctly conform the keypoint output format,
            if missing/invalid points will be filled by the mean coordiantes of the visible points
  """
  labels, points = tokens_to_regions(tokens, image_size, token_per_label=2)
  points = np.array(points)
  invalid = False  # Is this text a valid keypoint prediction

  # Convert label to integers
  for i, l in enumerate(labels):
    l = tokenizer.decode(l)
    try:
      l = int(l) - 1
      if not (0 <= l <= 2):
        invalid = True
        l = 0
    except ValueError:
      invalid = True
      l = 0
    labels[i] = l
  labels = np.array(labels)
  if np.sum(labels) == 0:
    # No visible points predicted
    return None, None, invalid

  # replace non visible point with mean so we do something non-crazy if the
  # GT turns out to be `visible`
  mean = np.mean(points[labels != 0], 0, keepdims=True)
  points[labels == 0] = mean

  if len(points) > 17:
    # Truncate if we generated extra for some reason
    invalid = True
    points = points[:17]
    labels = labels[:17]
  elif len(points) < 17:
    # Replace with mean if we generated too few points
    invalid = True
    mean = np.mean(points, 0, keepdims=True)
    n = 17 - len(points)
    points = np.concatenate([points, np.tile(mean, (n, 1))], 0)
    labels = np.concatenate([labels, np.zeros((n,), labels.dtype)])

  assert points.shape == (17, 2)
  return points, labels, invalid


def clean_mask(mask, min_size):
  """Remove connected components that have less than `min_size` pixels"""
  from scipy import ndimage
  label, n_obj = ndimage.measurements.label(mask)
  cleaned = None
  for c in range(1, n_obj+1):
    is_c = label == c
    if np.sum(is_c) > min_size:
      if cleaned is None:
        cleaned = is_c
      else:
        cleaned = np.logical_or(cleaned, is_c)
  return cleaned


def extract_segmentation_masks(img, segmention_mode="coarse_color") -> List[np.ndarray]:
  """Extract a list of binary segmentation masks from `img`"""
  if not np.issubdtype(img.dtype, np.integer):
    img = (img*255).astype(np.uint8)

  if segmention_mode == "any_pixel":
    # Assume there is only a single instance
    is_instance = img.mean(-1) > 30
    return [is_instance]

  elif segmention_mode == "coarse_color":
    # Find instances based on coarse-grained color detection, and then clean them for
    # extra/floating background pixels. Pretty slow, I think because `clean_mask` is slow
    w, h = img.shape[:2]
    img = np.array(img).reshape((-1, 3))  # [n_pixels, 3]

    img = img.astype(np.float64)
    means = img.mean(axis=-1)
    mean_diff = img - means[:, None]

    # Background pixels are black or nearly black
    background = means <= 30

    # First object pixels are gray/white, we allow gray since the VAE will often put gray
    # pixels around the white blobs it is supposed to predict
    # We detect such pixels if all RGB values are close to the mean
    first_obj = np.logical_and(np.logical_not(background), np.abs(mean_diff).sum(-1) < 100)
    used = np.logical_and(background, first_obj)  # Pixel already assigned
    out = []
    first_obj = clean_mask(first_obj, 10)
    if np.any(first_obj):
      out.append(first_obj)

    color = np.argmax(img, -1)
    for c in range(3):
      # Find pixels if each color they must have that color's value
      # be the largest RGB value be large then the mean by a reasonable margin
      candidate = np.logical_and(np.logical_not(used), color == c)
      color_map = np.logical_and(candidate, np.abs(mean_diff[:, c]) > 40)
      color_map = clean_mask(color_map, 10)
      if np.any(color_map):
        out.append(color_map)
        used = np.logical_and(used, color_map)
    return [x.reshape(w, h) for x in out]

  else:
    raise NotImplementedError()


def _resize(image: np.ndarray, target_size: Sequence[int],
            mode: Union[str, InterpolationMode]="bilinear", antialias=True):
  if isinstance(mode, str):
    mode = InterpolationMode(mode)
  if image.dtype == np.uint8:
    image = image / 255.0
  image = F.resize(torch.as_tensor(image.transpose((2, 0, 1))), target_size, antialias=antialias,
                   interpolation=mode)
  image = np.transpose(image.numpy().astype(np.float32), [1, 2, 0])
  return image


def resize_and_pad(image: np.ndarray, size) -> Tuple[np.ndarray, np.ndarray]:
  """Resize and pad `image` to `size` and returns a mask over pixels introduced by padding"""
  h, w = image.shape[:2]
  scale = size[0] / max(h, w)
  if scale != 1.0:
    scale_to = (int(h*scale), int(w*scale))
    image = _resize(image, scale_to)
  else:
    scale_to = (h, w)
  image_mask = np.zeros(size, dtype=np.bool)
  image_mask[:scale_to[0], :scale_to[1]] = True
  image = np.pad(
    image,
    [[0, size[0] - scale_to[0]],
     [0, size[1] - scale_to[1]],
     [0, 0]
     ]
  )
  return image, image_mask


def undo_image_preprocessing(image, original_size, mode="nearest", antialias=False):
  """Resize image generated from UnifiedIO to the size of `original_size`, this undoes
  the padding and down-scaling done in `preprocess_image`.

  By default, we use near-neighbor interpolation and not anti-aliasing since that makes the most
  sense for tasks involving non-natural images like segmentation and surface normals
  """
  h, w = original_size
  ratio = image.shape[0] / max(w, h)
  # undo the padding
  if h > w:
    image_rescale = image[:, :int(ratio*w)]
  else:
    image_rescale = image[:int(ratio*h), :]
  # Undo the scaling
  return _resize(np.copy(image_rescale), (h, w), mode=mode, antialias=antialias)


def preprocess_image(input_image, mask_region=None) -> Tuple[np.ndarray, np.ndarray]:
  """Preprocess an image for processing UnifiedIO

  :param input_image: image array in [h, w, 3] in float or uint8 format
  :param mask_region: Optional region to include in the image mask, used for image inpaintin
  :return: preprocessed image and image-patch mask
  """
  n_patches = 384//16
  if input_image is not None:
    original_size = input_image.shape
    input_image, image_mask = resize_and_pad(input_image, IMAGE_INPUT_SIZE)

    if mask_region is not None:
      region = mask_region / max(original_size[:2]) * max(input_image.shape[:2])
      x1, y1, x2, y2 = np.round(region).astype(np.int32)
      region_mask = np.ones_like(image_mask)
      region_mask[y1:y2, x1:x2] = 0
      image_mask = image_mask*region_mask

    # Convert mask over pixels to mask of image patches
    image_mask = _resize(
      np.expand_dims(image_mask, 2), [n_patches, n_patches],
      InterpolationMode.NEAREST, antialias=False
    )
    image_mask = image_mask.reshape((-1,)).astype(np.int32)
  else:
    if mask_region is not None:
      raise ValueError()
    # Masked, dummy values since this code does not support skipping the image
    input_image = np.zeros((384, 384, 3), np.float32)
    image_mask = np.zeros((n_patches*n_patches, ), dtype=np.int32)
  input_image = normalize_image(input_image)
  return input_image, image_mask


def preprocess_target_image(target_image) -> Tuple[np.ndarray, np.ndarray]:
  """Preprocess a target image for processing UnifiedIO

  :param target_image: image array in [h, w, 3] in float or uint8 format
  :return: preprocessed image and image-patch mask
  """
  n_patches = IMAGE_TARGET_SIZE[0]//16
  if target_image is not None:
    input_image, image_mask = resize_and_pad(target_image, IMAGE_TARGET_SIZE)

    # Convert mask over pixels to mask of image patches
    image_mask = _resize(
      np.expand_dims(image_mask, 2), [n_patches, n_patches],
      InterpolationMode.NEAREST, antialias=False
    )
    image_mask = image_mask.reshape((-1,)).astype(np.int32)
  else:
    input_image = np.zeros(IMAGE_TARGET_SIZE + [3], np.float32)
    image_mask = np.zeros((n_patches*n_patches, ), dtype=np.int32)
  input_image = input_image * 2 - 1  # VAE pre-processing
  return input_image, image_mask


BIAS = np.array([0.485, 0.456, 0.406])
SCALE = np.array([0.229, 0.224, 0.225])


def normalize_image(image) -> np.ndarray:
  """Pixel normalizing used by UnifiedIO"""
  image -= BIAS.reshape((1, 1, 3))
  image /= SCALE.reshape((1, 1, 3))
  return image

