from pydantic import BaseModel as ConfigBaseModel

class Config(ConfigBaseModel):
    # data
    base_dir = "/media/bisnu-sarkar/A82A0A732A0A3EB6/Data/birdclef-2023/"
    train_sound_dir = "/media/bisnu-sarkar/A82A0A732A0A3EB6/Data/birdclef-2023/train_audio/"
    path_train = base_dir + "train_metadata.csv"
    path_sample_submission = base_dir + "sample_submission.csv"
    sample_rate = 32_000
    # spec
    img_size = (128, 256)
    seconds = 5
    num_offset_max = 24
    min_duration = 0.5
    n_fft = 2048
    n_mels = img_size[0]
    hop_length = (seconds * sample_rate - n_fft) // (img_size[1] - 1) 
    center = False
    fmin = 500
    fmax = 12_500
    top_db = 80
    # output
    out_dir = "/media/bisnu-sarkar/A82A0A732A0A3EB6/Data/birdcleft_2023_images/mel_spec_images/"
    jpeg_quality = 100
    

cfg = Config()

print(cfg.sample_rate)