import argparse
import json
from os.path import exists
from PIL import Image, ImageDraw, ImageFont
from uio import runner
from uio.configs import CONFIGS
from uio import utils
from uio.runner import IMAGE_TAGGING
import numpy as np
import spacy
from absl import logging
from pathlib import Path
import warnings
# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
        "--model-size", type=str, choices = list(CONFIGS), help="The model size to use in the script."
  )
  parser.add_argument("--model-weights", type=str,required=True, help="Path to the .bin file containing the model weights.")
  parser.add_argument("--input-file", type=Path, required=True, help="Path to the input file containing image paths and prompts." )
  parser.add_argument("--classes-file", type=Path, required=True, help = "JSON file containing class names.")
  parser.add_argument("--output-file", type=Path,required=True, help="File to store the predicted classes.")
  parser.add_argument('--alternate-prompts', type=Path,required=False, help="Optional file containing alternate prompts.")

  args = parser.parse_args()
  input_file = args.input_file
  output_file = args.output_file
  tagging_prompts = None

  if args.alternate_prompts is not None:
    with args.alternate_prompts.open('r') as fp:
      tagging_prompts = [line.strip() for line in fp.readlines()]

  else:
    tagging_prompts = [IMAGE_TAGGING]

  model = runner.ModelRunner(args.model_size, args.model_weights)
  
  with args.classes_file.open('r') as fp:
    classes_dict = json.load(fp)
    classes = classes_dict['classes']


  lines = input_file.open('r').readlines()

  for line in lines:
    # ignore question for image classification
    image_path, _ = line.strip().split(":")
    logging.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
      image = np.array(img.convert('RGB'))

    for prompt in tagging_prompts:
      output = model.image_classification(image, answer_options=classes, prompt=prompt)
      output_text = output["text"]		
      
      with open(output_file, 'a') as of:
        # Log the question used by the model to the output
        of.write(f"{image_path}:{prompt}:{output_text}\n")


if __name__ == "__main__":
  main()
