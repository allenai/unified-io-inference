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
  parser.add_argument("vizwiz_file")
  parser.add_argument("image_dir")
  parser.add_argument("output_file")
  parser.add_argument("sample_count")

  args = parser.parse_args()
  output_file = args.output_file
  sample_count = int(args.sample_count)
  model = runner.ModelRunner(args.model_size, args.model_weights)

  logging.info(f"Reading: {args.vizwiz_file}")
  vizwiz_file = open(f"{args.vizwiz_file}")
  vizwiz_data = json.load(vizwiz_file)
  annotations = vizwiz_data['annotations']

  captions = []
  image_ids = []

  for anno in annotations:
    try:
#      print('ANNO: ' + str(anno))
      image_id = anno['image_id']
      # ...is_rejected': True, 'id': 2, 'text_detected': True
      if (not anno['is_rejected'] and anno['text_detected'] and image_id not in image_ids):
        print(str(image_id) + ': ' + anno['caption'])
        #TODO: Ex: VizWiz_test_00000000.jpg vs. train vs. val how best to switch?
        image_path = f"{args.image_dir}/VizWiz_train_{image_id:08d}.jpg"
        logging.info(f"Processing image: {image_path}")
        with Image.open(image_path) as img:
          image = np.array(img.convert('RGB'))
          output = model.vqa(image, "What does the image describe ?")
          output_text = output["text"]
          print(output_text)
          captions.append({"image_id": image_id, "caption": output_text})
          logging.info(f"Captions: {len(captions)}")
          image_ids.append(image_id)
        if len(image_ids) >= sample_count:
          break
    except AttributeError:
      print('AttributeError')

  logging.info(f"Writing: {output_file}")
  with open(output_file, 'a') as f:
    json.dump(captions, f)
  logging.info(f"Wrote: {output_file}")  

  quit()

# $ head /nas/gaia02/data/paper2023/vizwiz/data/annotations/train.json | cut -c -1000
# {"info": {"description": "This dataset contains crowdsourced captions of images from VizWiz datasets. This file contains the train partition.", "license": {"url": "https://creativecommons.org/licenses/by/4.0/", "name": "Attribution 4.0 International (CC BY 4.0)"}, "url": "https://vizwiz.org", "version": "VizWiz-Captions 1.0", "year": 2019, "contributor": "VizWiz-Captions Consortium", "date_created": "2019-12-23"}, "images": [{"file_name": "VizWiz_train_00000000.jpg", "vizwiz_url": "https://ivc.ischool.utexas.edu/VizWiz_visualization_img/VizWiz_train_00000000.jpg", "id": 0, "text_detected": true},
# "annotations": [{"captio": "A computer screen shows a repair prompt on the screen.", "image_id": 23431, "is_precanned": false, "is_rejected": false, "id": 117155, "text_detected": true}, {"caption": "a computer screen with a repair automatically pop up", "image_id": 23431, "is_precanned": false, "is_rejected": false, "id": 117156, "text_detected": true},


  ds = wds.WebDataset(args.webdataset_file)
  for sample in islice(ds, 0, int(args.sample_count)):
    item = json.loads(sample['json'])
    imageStream = io.BytesIO(sample['jpg'])
    imageFile = Image.open(imageStream)
    image = np.array(imageFile.convert('RGB'))
    output = model.vqa(image, "What does the image describe ?")
    output_text = output["text"]		
    logging.info(f"\n{sample['__key__']}\n{item['caption']}\n{output_text}\n")

# Write RES file:
# [{"image_id": 23431, "caption": "A blah blah..."},{...}]

    with open(output_file, 'a') as of:
      of.write(f"{sample['__key__']}\t{item['caption']}\t{output_text}\n")

if __name__ == "__main__":
  main()
