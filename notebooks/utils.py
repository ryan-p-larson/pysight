#! /usr/bin/env python3
# utils.py -- Shared by notebooks

# Add repository root to path for imports
import os, sys
repo_root = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.append(repo_root)

# Add image directory and images
img_dir = os.path.join(repo_root, 'tests', 'test_imgs')
img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]

# Models
model_dir = os.path.join(repo_root, 'pysight', 'models')
model_paths = [os.path.join(model_dir, model) 
               for model in os.listdir(model_dir)
               if (model.endswith(".xml") or model.endswith(".dat"))]


# Utils
import PIL.Image
import IPython.display
import numpy as np
from io import BytesIO

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))