import os
import shutil
import sys


import pandas as pd
import argparse

filename = "ISIC_2019_Training_GroundTruth.csv"

"""list_filename = []
list_labels = []
with open(filename) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
print("content: ",content[1])"""
def main(input_dir, output_dir):
    ############## use df
    df = pd.read_csv(filename)#, header=None)
    df = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 9]], axis = 1)
    print(df.head())
    
    mel_ouput_dir = os.path.join(output_dir, "mel")
    ben_output_dir = os.path.join(output_dir, "ben")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set the input and the output directory.')
    parser.add_argument("--input_dir", "-i", help="set input directory", type=str)
    parser.add_argument("--output_dir", "-o", help="set output directory", type=str)

    args = parser.parse_args()

    # Check for --width
    if args.input_dir and args.output_dir:
        import os
        if not os.path.exists(args.input_dir):
            print("Input folder does not exist" + args.input_dir)
            sys.exit(1)
        if not os.path.exists(args.output_dir):
            print("Output folder does not exist: " + args.output_dir)
            sys.exit(1)
        main(args.input_dir, args.output_dir)

# check size == 1024 x 1024 ?