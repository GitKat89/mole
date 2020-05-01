import os
import shutil
import sys


import pandas as pd
import argparse



def move_images(df, input_dir, mel_ouput_dir, ben_output_dir, copy_flag):
    

    file_function = shutil.move
    if copy_flag:
        file_function = shutil.copy
    
    for index, row in df.iterrows():
        image_path = ''
        try:
            #print(index, row)
            print("image name: ", row["image"])
            print("MEL- classified: ", row["MEL"])
            image_path = os.path.join(input_dir, str(row["image"] + ".jpg"))

            if row["MEL"] == 1:
                file_function(image_path, mel_ouput_dir)
            elif row["MEL"] == 0:
                file_function(image_path, ben_output_dir)
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
def main(input_dir, target_dir, copy_flag):
    ############## use df
    filename = "ISIC_2019_Training_GroundTruth.csv"

    df = pd.read_csv(filename) #, header=None)
    df = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 9]], axis = 1)
    #df = df[:20] # for testing purposes with only 20 images

    df_train = df[(df.index % 5 != 0) & (df.index % 101 != 0)]  # Excludes every 5th and every 101th row starting from 0
    df_val = df[(df.index % 5 == 0) & (df.index % 101 != 0)] # Selects every 5th row, excluding every nth modulo 101 row
    df_test = df[df.index % 101 == 0] #Selects every 101th row

    print(df.head())

    train_mel_path, train_ben_path = create_folder_structure(target_dir, "train")
    val_mel_path, val_ben_path = create_folder_structure(target_dir, "valid")
    test_path = os.path.join(target_dir, "test/mixed")
    create_folder(test_path)

    move_images(df_train, input_dir, train_mel_path, train_ben_path, copy_flag)
    move_images(df_val, input_dir, val_mel_path, val_ben_path, copy_flag)
    move_images(df_test, input_dir, test_path, test_path, copy_flag)
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set the input and the output directory.')
    parser.add_argument("--input_dir", "-i", help="set input directory", type=str)
    parser.add_argument("--output_dir", "-o", help="set output directory", type=str)
    parser.add_argument("--copy", "-c", help="use copy instead of move", type=bool, default=False)

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        import os
        if not os.path.exists(args.input_dir):
            print("Input folder does not exist" + args.input_dir)
            sys.exit(1)
        if not os.path.exists(args.output_dir):
            print("Output folder does not exist and will be created under: " + args.output_dir)
            create_folder(args.output_dir)
 
        main(args.input_dir, args.output_dir, args.copy)

