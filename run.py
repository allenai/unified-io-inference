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
  parser.add_argument("input_file")
  args = parser.parse_args()

  model = runner.ModelRunner(args.model_size, args.model_weights)
  input_file = open(args.input_file, 'r')
  lines = input_file.readlines()
  for line in lines:
    image_path, question = line.strip().split(":")
    print(image_path)
    print(question)
    with Image.open(image_path) as img:
      image = np.array(img.convert('RGB'))
      output = model.vqa(image, question)
      print(output["text"])


if __name__ == "__main__":
  main()
