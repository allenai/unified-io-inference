import argparse
import json
from os.path import exists
from PIL import Image, ImageDraw, ImageFont
from uio import runner
from uio.configs import CONFIGS
from uio import utils
import numpy as np
import spacy
from absl import logging
import warnings
from pathlib import Path
# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_size", choices=list(CONFIGS))
  parser.add_argument("model_weights")
  parser.add_argument("input_file")
  parser.add_argument("classes_file")
  parser.add_argument("output_file")

  args = parser.parse_args()
  input_file = open(args.input_file,'r')
  output_file = args.output_file
  
  model = runner.ModelRunner(args.model_size, args.model_weights)
  
  with open(args.classes_file,'r') as fp:
    classes_dict = json.load(fp)
    classes = classes_dict['classes']


  lines = input_file.readlines()

  for line in lines:
    # ignore question for image classification
    image_path, _ = line.strip().split(":")
    logging.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
      image = np.array(img.convert('RGB'))

    output = model.image_classification(image, answer_options=classes)
    output_text = output["text"]		
    
    with open(output_file, 'a') as of:
      # Log the question used by the model to the output
      of.write(f"{image_path}:{IMAGE_TAGGING}:{output_text}\n")


if __name__ == "__main__":
  main()

