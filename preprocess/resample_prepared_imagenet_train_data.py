import os
import shutil
import random
from tqdm import tqdm

RANDOM_SEED = 6
random.seed(RANDOM_SEED)

SAMPLED_MAJOR_CLASS_NUM = 800
MAJOR_CLASS_IMG_MAX_COUNT = 500
SAMPLED_MINOR_CLASS_NUM = 200
MINOR_CLASS_IMG_MAX_COUNT = 0
RESAMPLED_TRAIN_IMG_NUM = 50

IMAGENET_DATA_FOLDER = "../data/ImageNet/"
SAMPLED_IMAGENET_DATA_FOLDER = (
    "../data/Sampled_ImageNet_"
    f"{SAMPLED_MAJOR_CLASS_NUM}x{MAJOR_CLASS_IMG_MAX_COUNT}_"
    f"{SAMPLED_MINOR_CLASS_NUM}x{MINOR_CLASS_IMG_MAX_COUNT}_"
    f"Seed_{RANDOM_SEED}/"
)

SAMPLED_IMAGENET_TRAIN_DATA_FOLDER = os.path.join(
    SAMPLED_IMAGENET_DATA_FOLDER, "train"
)
RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER = os.path.join(
    SAMPLED_IMAGENET_DATA_FOLDER, f"resampled_{RESAMPLED_TRAIN_IMG_NUM}_train"
)

# Ensure the base folder for resampled train data exists and is empty
if os.path.exists(RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER):
    shutil.rmtree(RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER)
os.makedirs(RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER)

# Iterate over each class directory in the sampled train data folder
for class_dir in tqdm(os.listdir(SAMPLED_IMAGENET_TRAIN_DATA_FOLDER)):
    class_dir_path = os.path.join(
        SAMPLED_IMAGENET_TRAIN_DATA_FOLDER, class_dir
    )

    # Ensure the class directory exists and is a directory
    if os.path.isdir(class_dir_path):
        # Get all image file names in the class directory
        image_files = [
            f for f in os.listdir(class_dir_path)
            if os.path.isfile(os.path.join(class_dir_path, f))
        ]

        # Randomly sample image files
        sampled_image_files = random.sample(
            image_files, min(RESAMPLED_TRAIN_IMG_NUM, len(image_files))
        )

        # Create corresponding class directory in the resampled train data folder
        resampled_class_dir_path = os.path.join(
            RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER, class_dir
        )
        os.makedirs(resampled_class_dir_path, exist_ok=False)

        # Copy each sampled image to the new class directory
        for image_file in sampled_image_files:
            src_image_path = os.path.join(class_dir_path, image_file)
            dst_image_path = os.path.join(resampled_class_dir_path, image_file)
            shutil.copy(src_image_path, dst_image_path)

print(
    f"Done! The ImageNet training data has been resampled and saved to {RESAMPLED_IMAGENET_TRAIN_DATA_FOLDER}"
)
