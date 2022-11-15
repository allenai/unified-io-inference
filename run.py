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
  nlp = spacy.load("en_core_web_sm")

  lines = input_file.readlines()
  for line in lines:
    image_path, question = line.strip().split(":")
    logging.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
      image = np.array(img.convert('RGB'))
#ignore question
#/image-data/RTS2P7XB.jpg:What does the image describe?:a swamp is full of reeds that have partially bowed.
#/image-data/RTS2P7XB.jpg:What is in this image?:water.
      caption = model.vqa(image, "What does the image describe ?")
      j=[]
      for k,v in caption.items():
        type_v = type(v)
        try:
          j.append({json.dumps(k):json.dumps(v)})
        except:
          j.append({json.dumps(k):f"NOT SERIALIZABLE: {type_v}"})
      debug_output = json.dumps(j)
      logging.info((f"DEBUG CAPTION: {debug_output}")[0:1000])

      categorize = model.vqa(image, "What is in this image ?")
      j=[]
      for k,v in categorize.items():
        type_v = type(v)
        try:
          j.append({json.dumps(k):json.dumps(v)})
        except:
          j.append({json.dumps(k):f"NOT SERIALIZABLE: {type_v}"})
      debug_output = json.dumps(j)
      logging.info((f"DEBUG CATEGORIZE: {debug_output}")[0:1000])

      categorize_text = categorize["text"]
      caption_text = caption["text"]
      all_text = f"{categorize_text} {caption_text}"

      phrases = []
      current_text = ''

      for tok in caption_text.split(" "):
        if len(tok.strip()) > 0:
          t = tok.strip()
          doc = nlp(t)
          pos = str(doc[0].pos_)
          logging.info(f"{doc[0]} {pos}")

          if ("DET" == pos and '' == current_text) \
            or ("PRON" == pos and '' == current_text) \
            or "NOUN" == pos or "PROPN" == pos:
            current_text = f'{current_text} {doc[0]}'.strip()
          elif len(current_text) > 0:
            phrases.append(current_text)
            re_result = refexp(model, image, current_text)
            draw(img, re_result, current_text)
            current_text = ''

      if len(current_text) > 0:
        phrases.append(current_text)
        re_result = refexp(model, image, current_text)
        draw(img, re_result, current_text)
        current_text = ''         

      output = model.vqa(image, "Locate all objects in the image .")
      token = ''
      ref_tokens = []
      text = output["text"].replace("<"," <")

      for tok in text.split(" "):
        if len(tok)>10 and tok.startswith("<extra_id_"):
          ref_tokens.append(tok.strip())
        elif 2 < len(str(tok).strip()):
          token = tok.strip()
          logging.info(f"DEBUG token: {token}, extra_ids: {len(ref_tokens)}")
          ref_output = refexp(model, image, token)
          draw(img, ref_output, token)

#          for i in ref_tokens:
#            logging.info(f"SKIP: {i}")
#            ref_output = refexp(model, image, i)
          ref_tokens = []

      categorize_text = categorize["text"]
      caption_text = caption["text"]
      write(img, f"1: {caption_text}\n2: {categorize_text}")
      out_image_path = image_path + '.boxes.png'
      img.save(out_image_path)

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
      logging.info(f"Output: {output_text}\n\n")

def log(results):
    text = results["text"]
    logging.info(f"DEBUG text: {text}")    
    if "boxes" in results.keys() and len(results["boxes"]) > 0:
      box = results["boxes"][0]
      logging.info(f"BOX {box[0]}, {box[1]}, {box[2]}, {box[3]}")
      if len(results["boxes"]) > 1:
        logging.info(f"[...more boxes...]")

def draw(img, results, token):
    canvas = ImageDraw.Draw(img)
    if "boxes" in results.keys() and len(results["boxes"]) > 0:
      for box in results["boxes"]:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        shape = [(x1, y1), (x2, y2)]
        width, height = img.size
        w = 10 if width > 1000 else 5
        canvas.rectangle(shape, outline="red", width=w)
        text = str(results["text"])
        logging.info(f"DTEXT: {text} TOKEN: {token}")
        font_size = 80 if width > 1000 else 50
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        canvas.text((x1-1,y1-1), token, font=font, fill="white")
        canvas.text((x1-1,y1+1), token, font=font, fill="white")
        canvas.text((x1+1,y1-1), token, font=font, fill="white")
        canvas.text((x1+1,y1+1), token, font=font, fill="white")
        canvas.text((x1,y1), token, font=font, fill="red")

def write(img, text):
    logging.info(f"WTEXT: {text}")
    canvas = ImageDraw.Draw(img)
    width, height = img.size
    font_size = 80 if width > 3500 else 48 if width > 2000 else 24
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    x = 25
    y = height / 2
    canvas.text((x-1,y-1), text, font=font, fill="white")
    canvas.text((x-1,y+1), text, font=font, fill="white")
    canvas.text((x+1,y-1), text, font=font, fill="white")
    canvas.text((x+1,y+1), text, font=font, fill="white")
    canvas.text((x,y), text, font=font, fill="red")

def refexp(model, image, text):
    try:
      results = model.refexp(image, text)
      log(results)
      return results
    except ValueError as arg:
      logging.info(f"ERROR: {arg}")
      return {}

if __name__ == "__main__":
  main()


#Workbook example:
#'Which region does the text " {} " describe ?'
#sportsball=uio.refexp(soccer_img, "<extra_id_617>")
#To extract digit from extra_token
#          logging.info(f"TOKEN: {int(''.join(i for i in tok if i.isdigit()))}")
#          tokens.append(int(''.join(i for i in tok if i.isdigit())))
#      a, b = utils.tokens_to_regions(tokens, (384, 384))
#      logging.info(f"{str(a)}, {str(b)}")
