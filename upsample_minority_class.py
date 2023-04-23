import pandas as pd
import shutil
import os, pathlib
import cv2

img_stats = pd.read_csv("/media/bisnu-sarkar/A82A0A732A0A3EB6/Deep_learning/birdCLEF2021/Identify-bird-calls-in-soundscapes-BirdCLEF-2023-/img_stats.csv")
out_dir = "/media/bisnu-sarkar/A82A0A732A0A3EB6/Data/birdcleft_2023_images/mel_spec_images/"
counts = img_stats['label'].value_counts().sort_values(ascending=False)
threshold = 4

down_classes=[]

for idx,val in zip(counts.index, counts.values):
  if val<threshold:
    down_classes.append(idx)
    print(idx,val)

j=0
index_begin=124490
for c in down_classes:
      # get the dataframe for the current class
      class_df = img_stats.query("label==@c")
      print(class_df.shape[0])
      # # find number of samples to add
      num_up = threshold - class_df.shape[0]
      # # upsample the dataframe
      class_df_upsampled = class_df.sample(n=num_up, replace=True, random_state=40)
      print("Later : ", class_df_upsampled.shape[0])
      # # append the upsampled dataframe to the list
      last_offset = class_df_upsampled['offset'].max()
      print("Last offset :",last_offset)
     
      updated_df = []
      for index,row in class_df_upsampled.iterrows():
        last_offset+=1
        row['offset']=last_offset
        fname=row['filename'].split('.')[-2]
        fname= fname.split('_')[-2]
        fname = fname.split('/')[-1]
        fname = f"{fname}_{last_offset}.jpeg"
        rec_dir = out_dir + row["filename"].split('/')[0]
        src_img = out_dir + row["filename"]
        dest_img = os.path.join(rec_dir, fname)
        print(src_img)
        print(dest_img)
        shutil.copy(src_img, dest_img)
        row.update({
            "offset":last_offset,
            "filename": "/".join(pathlib.Path(dest_img).parts[-2:]),
        })
        updated_df.append(row)
      updated_df = pd.DataFrame(updated_df)
      print(updated_df)
      img_stats = pd.concat([img_stats, updated_df],ignore_index=True)
     
img_stats.to_csv("img_stats_with_upsample.csv", index=False)      
print(img_stats.tail(10))