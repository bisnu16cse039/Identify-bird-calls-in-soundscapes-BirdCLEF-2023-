import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from datetime import datetime
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel as ConfigBaseModel
from train_config import TrainConfig
import tensorflow as tf
print("tensorflow:", tf.__version__)
import keras_cv
print("keras_cv:", keras_cv.__version__)
import tensorflow_io as tfio
print("tfio:", tfio.__version__)
from sklearn.model_selection import train_test_split

# strategy = tf.distribute.MirroredStrategy()
# print("Strategy:", strategy)
# print("Number of replicas:", strategy.num_replicas_in_sync)

if __name__ == "__main__":
    cfg= TrainConfig()
    data = pd.read_csv(cfg.path_data)
    data["path_img"] = cfg.dataset_dir + data["filename"]
    if cfg.debug:
        data = data.iloc[:100]
    data
