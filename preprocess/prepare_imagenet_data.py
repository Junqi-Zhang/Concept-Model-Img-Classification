import os
import shutil
import random
from tqdm import tqdm
from imagenet_classes import IMAGENET2012_CLASSES

SAMPLED_MAJOR_CLASS_NUM = 200
MAJOR_CLASS_IMG_MAX_COUNT = 500
SAMPLED_MINOR_CLASS_NUM = 50
MINOR_CLASS_IMG_MAX_COUNT = 100

IMAGENET_DATA_FOLDER = "../data/ImageNet/"
SAMPLED_IMAGENET_DATA_FOLDER = "../data/Sampled_ImageNet/"

########## 清空采样结果文件夹 Start ##########
if os.path.exists(SAMPLED_IMAGENET_DATA_FOLDER):
    shutil.rmtree(SAMPLED_IMAGENET_DATA_FOLDER)
os.makedirs(SAMPLED_IMAGENET_DATA_FOLDER)
########## 清空采样结果文件夹 Start ##########

    
########## 类别采样 Start ##########
class_codes = list(IMAGENET2012_CLASSES.keys())
assert len(class_codes) == 1000

sampled_class_codes = random.sample(
    class_codes, SAMPLED_MAJOR_CLASS_NUM+SAMPLED_MINOR_CLASS_NUM)
sampled_major_class_codes = sampled_class_codes[:SAMPLED_MAJOR_CLASS_NUM]
sampled_minor_class_codes = sampled_class_codes[-SAMPLED_MINOR_CLASS_NUM:]
########## 类别采样 Start ##########


########## 类别采样正确性检查 Start ##########
for i, code in enumerate(sampled_major_class_codes):
    assert sampled_class_codes[i] == code

for i, code in enumerate(sampled_minor_class_codes):
    assert sampled_class_codes[SAMPLED_MAJOR_CLASS_NUM+i] == code

for code in sampled_minor_class_codes:
    assert code not in sampled_major_class_codes

for code in sampled_major_class_codes:
    assert code not in sampled_minor_class_codes
########## 类别采样正确性检查 FINISH ##########


########## train data 采样 Start ##########

sampled_train_img_num = 0

# 设定每个类别的采样数量
sample_num_per_class = dict()
for code in sampled_major_class_codes:
    sample_num_per_class[code] = MAJOR_CLASS_IMG_MAX_COUNT
for code in sampled_minor_class_codes:
    sample_num_per_class[code] = MINOR_CLASS_IMG_MAX_COUNT

# 创建每个类别的文件夹
for code in sampled_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "train", code))

# 采样 train data
all_train_img_file_names = os.listdir(os.path.join(IMAGENET_DATA_FOLDER, "train"))
random.shuffle(all_train_img_file_names) # shuffle 全部训练图片的顺序

# all_train_img_file_names = all_train_img_file_names[:1000] # For Debug

for i, file_name in tqdm(enumerate(all_train_img_file_names), total=len(all_train_img_file_names)):
    
    code = file_name.split("_")[0] # 'n03498962_56993_n03498962.JPEG'
    
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
########## train data 采样 FINISH ##########


########## val data 按类抽取 Start ##########
# 抽取三个 validation set
sampled_val_img_num = 0
sampled_major_val_img_num = 0
sampled_minor_val_img_num = 0

# 创建每个类别的文件夹
for code in sampled_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "val", code))
for code in sampled_major_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "major_val", code)) 
for code in sampled_minor_class_codes:
    os.makedirs(os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "minor_val", code))

# 按类抽取 val data
all_val_img_file_names = os.listdir(os.path.join(IMAGENET_DATA_FOLDER, "val"))

for i, file_name in tqdm(enumerate(all_val_img_file_names), total=len(all_val_img_file_names)):
    
    code = file_name.split("_")[-1].split(".")[0] # 'ILSVRC2012_val_00000001_n01751748.JPEG'
    
    if code not in sampled_class_codes:
        continue
    
    src_img_file_path = os.path.join(IMAGENET_DATA_FOLDER, "val", file_name)
    dst_folder = os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "val", code)
    shutil.copy2(src_img_file_path, dst_folder)
    sampled_val_img_num += 1
    
    if code in sampled_major_class_codes:
        dst_folder = os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "major_val", code)
        shutil.copy2(src_img_file_path, dst_folder)
        sampled_major_val_img_num += 1
    
    if code in sampled_minor_class_codes:
        dst_folder = os.path.join(SAMPLED_IMAGENET_DATA_FOLDER, "minor_val", code)
        shutil.copy2(src_img_file_path, dst_folder)
        sampled_minor_val_img_num += 1
        
print(f"Finish Sampling {sampled_val_img_num} val images in total,")
print(f"in which {sampled_major_val_img_num} are in major classes,")
print(f"and {sampled_minor_val_img_num} are in minor classes.")
    
########## val data 按类抽取 FINISH ##########
