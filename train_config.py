from pydantic import BaseModel as ConfigBaseModel
from datetime import datetime
import os
class TrainConfig(ConfigBaseModel):
    ## general
    run_ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    debug = False
    model_name = "dev-b0-v8"
    test_size = 0.2
    seed = 887
    fit_verbose = 1 if (os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == "Interactive") else 2
    ## data
    dataset_dir = "/kaggle/input/ds-bc23-image-creation-128-x-256/train/"
    path_data = "/kaggle/input/ds-bc23-image-creation-128-x-256/img_stats.csv"
    label = "label"
    n_label = 264
    img_size = (128, 256)
    channels = 1
    img_shape = (*img_size, channels)
    ## model
    base_model_weights = "imagenet"
    dropout = 0.20
    ## training
    label_smoothing = 0.05
    shuffle_size = 1028
    steps_per_epoch = 300
    batch_size = 128  # 16 * strategy.num_replicas_in_sync
    valid_batch_size = batch_size
    epochs = 30
    patience = 4
    monitor = "val_loss"  # val_loss
    monitor_mode = "auto"
    lr = 1e-3
    ## aug
    aug_proba = 0.8