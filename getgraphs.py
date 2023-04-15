
import os
import pickle as pkl
import sys
import numpy as np


#import from current project directories
import datasets


#SETUP
NUM_JOBS = 100
JOB_ARRAY_NUMBER = 0

df = datasets.bigvul(minimal=False)
#print(df)