import os
import shutil
import sys


import pandas as pd
import argparse

filename = "ISIC_2019_Training_GroundTruth.csv"


def move_images(df, input_dir, mel_ouput_dir, ben_output_dir):
    for index, row in df.iterrows():
        image_path = ''
        try:
            #print(index, row)
            print("image name: ", row["image"])
            print("MEL- classified: ", row["MEL"])
            image_path = os.path.join(input_dir, str(row["image"] + ".jpg"))

            if row["MEL"] == 1:
                shutil.move(image_path, mel_ouput_dir)
            elif row["MEL"] == 0:
                shutil.move(image_path, ben_output_dir)
        except IOError:
            print("File not accessible " + image_path)

def create_folder_structure(target_dir, dir_name):
   
    parent_dir = os.path.join(target_dir,dir_name)
    create_folder(parent_dir)

    mel_ouput_dir = os.path.join(parent_dir, "melanoma")
    create_folder(mel_ouput_dir)
    
    ben_output_dir = os.path.join(parent_dir, "benigne")
    create_folder(ben_output_dir)

    return mel_ouput_dir,ben_output_dir

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


"""list_filename = []
list_labels = []
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
print("content: ",content[1])"""
def main(input_dir, target_dir):
    ############## use df
    df = pd.read_csv(filename)#, header=None)
    df = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 9]], axis = 1)
    #df = df[:20] # for testing purposes with only 20 images

    df_train = df[df.index % 5 != 0]  # Excludes every 5th row starting from 0
    df_val = df[df.index % 5 == 0]

    print(df.head())

    train_mel_dir, train_ben_dir = create_folder_structure(target_dir, "train")
    val_mel_dir, val_ben_dir = create_folder_structure(target_dir, "valid")

    move_images(df_train, input_dir, train_mel_dir, train_ben_dir)
    move_images(df_val, input_dir, val_mel_dir, val_ben_dir)
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set the input and the output directory.')
    parser.add_argument("--input_dir", "-i", help="set input directory", type=str)
    parser.add_argument("--output_dir", "-o", help="set output directory", type=str)

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        import os
        if not os.path.exists(args.input_dir):
            print("Input folder does not exist" + args.input_dir)
            sys.exit(1)
        if not os.path.exists(args.output_dir):
            print("Output folder does not exist: " + args.output_dir)
            sys.exit(1)
        main(args.input_dir, args.output_dir)

