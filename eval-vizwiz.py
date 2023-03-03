from vizwiz_api.vizwiz import VizWiz
from vizwiz_eval_cap.eval import VizWizEvalCap
import matplotlib.pyplot as plt
import numpy as np

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
annFile = '/input/train.json'
resFile = '/output/vizwiz-train100-captions.json'
vizwiz = VizWiz(annFile, ignore_rejected=True, ignore_precanned=True)
vizwizRes = vizwiz.loadRes(resFile)
vizwizEval = VizWizEvalCap(vizwiz, vizwizRes)
vizwizEval.evaluate()
for metric, score in vizwizEval.eval.items():
    print('%s: %.3f'%(metric, score))
