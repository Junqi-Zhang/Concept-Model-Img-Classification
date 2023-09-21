import os
import shutil
from tqdm import tqdm
from imagenet_classes import IMAGENET2012_CLASSES

IMAGENET_DATA_FOLDER = "../data/ImageNet/"
IMAGENET_VAL_DATA_FOLDER = ("../data/ImageNet_Val/")

########## Clear the sample result folder Start ##########
if os.path.exists(IMAGENET_VAL_DATA_FOLDER):
    shutil.rmtree(IMAGENET_VAL_DATA_FOLDER)
os.makedirs(IMAGENET_VAL_DATA_FOLDER)
########## Clear the sample result folder Finish ##########

class_codes = list(IMAGENET2012_CLASSES.keys())
assert len(class_codes) == 1000

########## Extract val data by class Start ##########
# Extract three validation sets
sampled_val_img_num = 0

# Create a folder for each category
for code in class_codes:
    os.makedirs(os.path.join(IMAGENET_VAL_DATA_FOLDER, code))

# Extract val data by class
all_val_img_file_names = os.listdir(os.path.join(IMAGENET_DATA_FOLDER, "val"))

for i, file_name in tqdm(enumerate(all_val_img_file_names), total=len(all_val_img_file_names)):

    # 'ILSVRC2012_val_00000001_n01751748.JPEG'
    code = file_name.split("_")[-1].split(".")[0]

    src_img_file_path = os.path.join(IMAGENET_DATA_FOLDER, "val", file_name)
    dst_folder = os.path.join(IMAGENET_VAL_DATA_FOLDER, code)
    shutil.copy2(src_img_file_path, dst_folder)
    sampled_val_img_num += 1

print(f"Finish Sampling {sampled_val_img_num} val images in total.")
########## Extract val data by class Finish ##########
