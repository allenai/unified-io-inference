from vizwiz_api.vizwiz import VizWiz
from vizwiz_eval_cap.eval import VizWizEvalCap
import matplotlib.pyplot as plt
#import skimage.io as io
import numpy as np
#import pylab
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
#In [2]:
# download Stanford models
#!bash get_stanford_models.sh # for Windows with Windows Subsystem for Linux (WSL) installed
#!./get_stanford_models.sh   # for Linux / MacOS
#Found Stanford CoreNLP.
#In [3]:
# file containing groundtruth
#annFile = './annotations/val.json'
annFile = '/root/vizwiz/vizwiz-caption/annotations/val.json'
#annFile = '/input/train.json'
#resFile = './results/fake_caption_val.json'
resFile = '/root/vizwiz/vizwiz-caption/results/fake_caption_val.json'
#In [4]:
# initialize VizWiz API for groundtruth 
vizwiz = VizWiz(annFile, ignore_rejected=True, ignore_precanned=True)
#loading annotations into memory...
#Done (t=0.10s)
#creating index...
#index created! imgs = 7750, anns = 33145
#In [5]:
# load generated results
vizwizRes = vizwiz.loadRes(resFile)
#Loading and preparing results...
#DONE (t=0.01s)
#creating index...
#index created! imgs = 7750, anns = 7750
#In [6]:
# create VizWizEval object by taking groundtruth and generated results
vizwizEval = VizWizEvalCap(vizwiz, vizwizRes)
#In [7]:
# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
vizwizEval.evaluate()
#tokenization...
#setting up scorers...
#computing Bleu score...
#{'testlen': 86410, 'reflen': 83011, 'guess': [86410, 78868, 71335, 63802], 'correct': [31457, 12531, 8925, 7741]}
#ratio: 1.0409463806001489
#Bleu_1: 0.364
#Bleu_2: 0.241
#Bleu_3: 0.193
#Bleu_4: 0.172
#computing Rouge score...
#ROUGE_L: 0.289
#computing CIDEr score...
#CIDEr: 0.318
#computing SPICE score...
#SPICE: 0.060
#computing METEOR score...
#METEOR: 0.120
#In [8]:
# print output evaluation scores
for metric, score in vizwizEval.eval.items():
    print('%s: %.3f'%(metric, score))
