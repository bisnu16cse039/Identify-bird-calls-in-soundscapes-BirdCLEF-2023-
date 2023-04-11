import os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import librosa
print("librosa:", librosa.__version__)
import tensorflow as tf
print("tensorflow:", tf.__version__)
import cv2
print("opencv:", cv2.__version__)
from IPython.display import Audio
from config import Config
from joblib import delayed, Parallel

def get_duration(rec):
    return librosa.get_duration(path=rec["path_ogg"])

def get_duration_df(df):
    return df.apply(get_duration, axis=1)

def normalize_img(img):
    """Normalize to uint8 image range"""
    assert img.ndim == 2, "unexpected dimension"
    v_min, v_max = np.min(img), np.max(img)
    return ((img - v_min) / (v_max - v_min) * 255).astype('uint8')



def get_mel_spec_db(path_ogg, offset):
    """Get dB scaled mel power spectrum"""
    required_len = cfg.seconds * cfg.sample_rate
    sig, dr = librosa.load(path=path_ogg, sr=cfg.sample_rate, offset=(offset * cfg.seconds), duration=cfg.seconds)
    sig = np.concatenate([sig, np.zeros((required_len - len(sig)), dtype=sig.dtype)])
    mel_spec = librosa.feature.melspectrogram(
        y=sig, 
        hop_length=cfg.hop_length,
        sr=cfg.sample_rate, 
        n_fft=cfg.n_fft, 
        n_mels=cfg.n_mels,
        center=cfg.center,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=cfg.top_db)
    return mel_spec_db

def process_record(rec):
    """Process a single record"""
    rec_dir = cfg.out_dir + rec.primary_label
    os.makedirs(rec_dir, exist_ok=True)
    stats = []
    base_stat = {"label": rec.label, "orig_filename": rec.filename}
    for offset in range(rec.num_offset):
        mel_spec_db = get_mel_spec_db(rec.path_ogg, offset=offset)
        img = normalize_img(mel_spec_db)
        fname = f"{pathlib.Path(rec.filename).stem}_{offset}.jpeg"
        path_img = os.path.join(rec_dir, fname)
        ret = cv2.imwrite(path_img, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        stat = base_stat.copy()
        stat.update({
            "offset": offset,
            "ret": ret,
            "filename": "/".join(pathlib.Path(path_img).parts[-2:]),
        })
        stats.append(stat)
    return pd.DataFrame(stats)


def process_data(data):
    """Process dataframe"""
    errors = []
    l_stats = []
    for rec in data.itertuples():
        try: 
            stats = process_record(rec)
            l_stats.append(stats)
        except Exception as err:
            print(f"Error reading {rec.filename}: {str(err)}")
            errors.append((rec.filename, str(err)))
    return l_stats, errors

if __name__ == "__main__":
    cfg= Config()
    data = pd.read_csv(cfg.path_train)
    data["path_ogg"] = cfg.train_sound_dir + data["filename"]

    sample_submission = pd.read_csv(cfg.path_sample_submission)
    labels = sample_submission.columns[1:].to_list()
    assert labels == sorted(labels), "labels are not sorted"
    label_encoder = pd.Series(np.arange(len(labels)), index=labels)
    data["label"] = data["primary_label"].map(label_encoder)

    durations = Parallel(n_jobs=os.cpu_count(), verbose=1, backend='multiprocessing')(
        delayed(get_duration_df)(sub) 
        for sub in np.array_split(data, os.cpu_count())
    )
    data["duration"] = pd.concat(durations)

    data["num_offset"] = (1 + (data["duration"] - cfg.min_duration) // cfg.seconds).astype('int')
    data["num_offset"] = data["num_offset"].clip(upper=cfg.num_offset_max)

    orig_out_dir = cfg.out_dir
    print("original output directory:", orig_out_dir)
    #cfg.out_dir = "/kaggle/temp/"
    results = Parallel(n_jobs=os.cpu_count(), verbose=1, backend='multiprocessing')(
    delayed(process_data)(sub) for sub in np.array_split(data, os.cpu_count())
    )

    errors = [x for r in results for x in r[1]]
    img_stats = [x for r in results for x in r[0]]
    if len(img_stats):
        img_stats = pd.concat(img_stats).reset_index(drop=True)
    img_stats











