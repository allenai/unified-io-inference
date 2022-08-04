import argparse
from os.path import exists

from PIL import Image

from uio import runner
from uio.configs import CONFIGS
import numpy as np

from absl import logging
import warnings

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_size", choices=list(CONFIGS))
  parser.add_argument("model_weights")
  args = parser.parse_args()

  if not exists("dbg_img.png"):
    logging.info("Downloading image")
    import urllib.request
    urllib.request.urlretrieve(
      "https://farm2.staticflickr.com/1362/1261465554_95741e918b_z.jpg",
      filename="dbg_img.png")

  model = runner.ModelRunner(args.model_size, args.model_weights)
  with Image.open("dbg_img.png") as img:
    image = np.array(img)
  output = model.vqa(image, "What color is the sofa?")
  print(output["text"])  # Should print `green`


if __name__ == "__main__":
  main()