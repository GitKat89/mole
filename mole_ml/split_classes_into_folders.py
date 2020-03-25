import os
import shutil

import pandas as pd
filename = "ISIC_2019_Training_GroundTruth.csv"

"""list_filename = []
list_labels = []
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
print("content: ",content[1])"""
############## use df
df = pd.read_csv(filename)#, header=None)
df = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 9]], axis = 1)
print(df.head())
READ_DIR = "./raw_data/"
MEL_OUT_DIR = READ_DIR + "mel"
BEN_OUT_DIR = READ_DIR + "ben"
for index, row in df.iterrows():
    #print(index, row)
    print("image name: ", row["image"])
    print("MEL- classified: ", row["MEL"])
    if row["MEL"] == 1:
        shutil.move(READ_DIR + str(row["image"] + ".jpg"), MEL_OUT_DIR)
    elif row["MEL"] == 0:
        shutil.move(READ_DIR + str(row["image"] + ".jpg"), BEN_OUT_DIR)







# check size == 1024 x 1024 ?