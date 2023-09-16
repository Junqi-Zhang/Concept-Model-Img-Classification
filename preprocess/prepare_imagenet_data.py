import os
import shutil
import random
from tqdm import tqdm
from imagenet_classes import IMAGENET2012_CLASSES

RANDOM_SEED = 6
random.seed(RANDOM_SEED)

SAMPLED_MAJOR_CLASS_NUM = 500
MAJOR_CLASS_IMG_MAX_COUNT = 1000
SAMPLED_MINOR_CLASS_NUM = 500
MINOR_CLASS_IMG_MAX_COUNT = 5

IMAGENET_DATA_FOLDER = "../data/ImageNet/"
SAMPLED_IMAGENET_DATA_FOLDER = (
    "../data/Sampled_ImageNet_"
    f"{SAMPLED_MAJOR_CLASS_NUM}x{MAJOR_CLASS_IMG_MAX_COUNT}_"
    f"{SAMPLED_MINOR_CLASS_NUM}x{MINOR_CLASS_IMG_MAX_COUNT}_"
    f"Seed_{RANDOM_SEED}/"
)

########## Clear the sample result folder Start ##########
if os.path.exists(SAMPLED_IMAGENET_DATA_FOLDER):
    shutil.rmtree(SAMPLED_IMAGENET_DATA_FOLDER)
os.makedirs(SAMPLED_IMAGENET_DATA_FOLDER)
########## Clear the sample result folder Finish ##########

########## Class sampling Start ##########
class_codes = list(IMAGENET2012_CLASSES.keys())
assert len(class_codes) == 1000

sampled_class_codes = random.sample(
    class_codes, SAMPLED_MAJOR_CLASS_NUM+SAMPLED_MINOR_CLASS_NUM
)
sampled_major_class_codes = sampled_class_codes[:SAMPLED_MAJOR_CLASS_NUM]
sampled_minor_class_codes = sampled_class_codes[-SAMPLED_MINOR_CLASS_NUM:]
########## Class sampling Finish ##########

########## Class sampling correctness check Start ##########
for i, code in enumerate(sampled_major_class_codes):
    assert sampled_class_codes[i] == code

for i, code in enumerate(sampled_minor_class_codes):
    assert sampled_class_codes[SAMPLED_MAJOR_CLASS_NUM+i] == code

for code in sampled_minor_class_codes:
    assert code not in sampled_major_class_codes

for code in sampled_major_class_codes:
    assert code not in sampled_minor_class_codes
########## Class sampling correctness check Finish ##########

########## Train data sampling Start ##########

sampled_train_img_num = 0

# Set the number of samples for each category
sample_num_per_class = dict()
for code in sampled_major_class_codes:
    sample_num_per_class[code] = MAJOR_CLASS_IMG_MAX_COUNT
for code in sampled_minor_class_codes:
    sample_num_per_class[code] = MINOR_CLASS_IMG_MAX_COUNT

# Create a folder for each category
for code in sampled_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "train", code))

# Sample train data
all_train_img_file_names = os.listdir(
    os.path.join(IMAGENET_DATA_FOLDER, "train")
)
# Shuffle the order of all training images
random.shuffle(all_train_img_file_names)

# all_train_img_file_names = all_train_img_file_names[:1000] # For Debug

for i, file_name in tqdm(enumerate(all_train_img_file_names), total=len(all_train_img_file_names)):

    code = file_name.split("_")[0]  # 'n03498962_56993_n03498962.JPEG'

    if code not in sample_num_per_class:
        continue

    # print(code, sample_num_per_class[code])
    src_img_file_path = os.path.join(IMAGENET_DATA_FOLDER, "train", file_name)
    dst_folder = os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "train", code)
    shutil.copy2(src_img_file_path, dst_folder)
    sampled_train_img_num += 1

    sample_num_per_class[code] -= 1
    if sample_num_per_class[code] == 0:
        sample_num_per_class.pop(code)
    if len(sample_num_per_class) == 0:
        break

print(f"Finish Sampling {sampled_train_img_num} train images in total.")
########## Train data sampling Finish ##########

########## Extract val data by class Start ##########
# Extract three validation sets
sampled_val_img_num = 0
sampled_major_val_img_num = 0
sampled_minor_val_img_num = 0

# Create a folder for each category
for code in sampled_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "val", code))
for code in sampled_major_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "major_val", code))
for code in sampled_minor_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "minor_val", code))

# Extract val data by class
all_val_img_file_names = os.listdir(os.path.join(IMAGENET_DATA_FOLDER, "val"))

for i, file_name in tqdm(enumerate(all_val_img_file_names), total=len(all_val_img_file_names)):

    # 'ILSVRC2012_val_00000001_n01751748.JPEG'
    code = file_name.split("_")[-1].split(".")[0]

    if code not in sampled_class_codes:
        continue

    src_img_file_path = os.path.join(IMAGENET_DATA_FOLDER, "val", file_name)
    dst_folder = os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "val", code)
    shutil.copy2(src_img_file_path, dst_folder)
    sampled_val_img_num += 1

    if code in sampled_major_class_codes:
        dst_folder = os.path.join(
            SAMPLED_IMAGENET_DATA_FOLDER, "major_val", code
        )
        shutil.copy2(src_img_file_path, dst_folder)
        sampled_major_val_img_num += 1

    if code in sampled_minor_class_codes:
        dst_folder = os.path.join(
            SAMPLED_IMAGENET_DATA_FOLDER, "minor_val", code
        )
        shutil.copy2(src_img_file_path, dst_folder)
        sampled_minor_val_img_num += 1

print(f"Finish Sampling {sampled_val_img_num} val images in total,")
print(f"in which {sampled_major_val_img_num} are in major classes,")
print(f"and {sampled_minor_val_img_num} are in minor classes.")
########## Extract val data by class Finish ##########
