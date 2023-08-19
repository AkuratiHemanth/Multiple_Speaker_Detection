''' Importing essential libraries such as TensorFlow for machine learning, file handling tools like os and shutil, 
numerical computations with numpy, high-level neural networks API keras, path manipulation with pathlib, 
content display via IPython.display, and system command execution via subprocess. 📚💻🔧'''

import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess


'''Copying the "speaker-recognition-dataset" directory 
from the input location to the current directory using the command !cp -r "../input/speaker-recognition-dataset" ./. 📂🔁'''

!cp -r "../input/speaker-recognition-dataset" ./



