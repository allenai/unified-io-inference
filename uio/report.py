import argparse
import json
from os.path import exists
from PIL import Image
from uio import runner
from uio.configs import CONFIGS
from uio import utils
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
  logging.info(f"Model: {args.model_size}")
  input_file = open(args.input_file, 'r')
  logging.info(f"Input file: {args.input_file}")
  output_file = f"{args.input_file}.{args.model_size}.results.txt"
  logging.info(f"Output file: {output_file}")

  lines = input_file.readlines()
  for line in lines:
    image_path, question = line.strip().split(":")
    logging.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
      image = np.array(img.convert('RGB'))
      output = model.vqa(image, question)
      token = ''
      ref_tokens = []
      for tok in output["text"].split(" "):
        if len(tok)>10 and tok.startswith("<extra_id_"):
          ref_tokens.append(tok)
        else:
          token = tok
#          logging.info(f"TOKEN: {int(''.join(i for i in tok if i.isdigit()))}")
#          tokens.append(int(''.join(i for i in tok if i.isdigit())))
          break

      logging.info(f"DEBUG REF_TOKEN COUNT: {len(ref_tokens)} {token}")

#'Which region does the text " {} " describe ?'
#sportsball=uio.refexp(soccer_img, "<extra_id_617>")

      for i in ref_tokens():
        ref_output = model.vqa(image, f"Which region does the text {i} describe ?")
        text = ref_output["text"]
        logging.info(f"{text}")
        box = ref_output["boxes"][0]
        logging.info(f"{json.dumps(box)}")

#      a, b = utils.tokens_to_regions(tokens, (384, 384))
#      logging.info(f"{str(a)}, {str(b)}")

      j=[]
      for k,v in output.items():
        type_v = type(v)
        try:
          j.append({json.dumps(k):json.dumps(v)})
        except:
          j.append({json.dumps(k):f"NOT SERIALIZABLE: {type_v}"})

      debug_output = json.dumps(j)
      logging.info((f"DEBUG: {debug_output}")[0:1000])

      output_text = output["text"]
      with open(output_file, 'a') as of:
        of.write(f"{image_path}:{question}:{output_text}\n")
      logging.info(f"Output: {output_text}")


if __name__ == "__main__":
  main()
