import argparse
import io
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
import webdataset as wds
from itertools import islice
from pathlib import Path
# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_size", choices=list(CONFIGS))
  parser.add_argument("model_weights")
  parser.add_argument("webdataset_file")
  parser.add_argument("output_file")
  parser.add_argument("sample_count")

  args = parser.parse_args()
  output_file = args.output_file
  model = runner.ModelRunner(args.model_size, args.model_weights)

  logging.info(f"Reading: {args.webdataset_file}")
  ds = wds.WebDataset(args.webdataset_file)
  for sample in islice(ds, 0, int(args.sample_count)):
    item = json.loads(sample['json'])
    imageStream = io.BytesIO(sample['jpg'])
    imageFile = Image.open(imageStream)
    image = np.array(imageFile.convert('RGB'))
    output = model.vqa(image, "What does the image describe ?")
    output_text = output["text"]		
    logging.info(f"\n{sample['__key__']}\n{item['caption']}\n{output_text}\n")

    with open(output_file, 'a') as of:
      of.write(f"{sample['__key__']}\t{item['caption']}\t{output_text}\n")

if __name__ == "__main__":
  main()
