import sys
sys.path.append('../')
from pathlib import Path
import json
from os import name
from preprocessor import Preprocessor
from tools import *
from preprocess import *

basedir = Path(__file__).resolve().parents[2]