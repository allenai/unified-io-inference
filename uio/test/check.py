from functools import partial
from jax import grad, lax
import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
from torchvision.io import read_image
import urllib.request
import spacy
from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from absl import logging
logging.set_verbosity(logging.INFO)
import utils
import runner
uio = runner.ModelRunner("xl", "xl.bin")

nlp = spacy.load("en_core_web_sm")
#a soccer player getting ready to control the ball.
doc = nlp("soccer players")
print(f'TAG: {doc[0].tag_}, POS: {doc[0].pos_} {str(doc[0])}')
print(f'TAG: {doc[1].tag_}, POS: {doc[1].pos_} {str(doc[1])}')

doc = nlp("a soccer player getting ready to control the ball.")
for item in doc:
  print(f'{str(item)} TAG: {item.tag_}, POS: {item.pos_}')

#def load_image_from_url(url):
#    with urllib.request.urlopen(url) as f:
#        img = Image.open(f)
#        return np.array(img)
#hotel_img = load_image_from_url('https://farm2.staticflickr.com/1362/1261465554_95741e918b_z.jpg')
#tennis_img = load_image_from_url('https://farm9.staticflickr.com/8313/7954229658_03f8e8d855_z.jpg')
#penguin_img = load_image_from_url('https://i.stack.imgur.com/z9vLx.jpg')
#uio.caption(hotel_img)["text"]
