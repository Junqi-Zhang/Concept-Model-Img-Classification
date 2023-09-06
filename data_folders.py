import os
from collections import OrderedDict


CALTECH_256 = OrderedDict(
    {
        "num_classes": 256,
        "train_folder_path": os.path.join("./data/", "Caltech-256"),
        "val_folder_path": os.path.join("./data/", "Caltech-256")
    }
)

SAMPLED_IMAGENET_200X1000_200X25_SEED_6 = OrderedDict(
    {
        "num_classes": 250,
        "train_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_200x25_Seed_6", "train"),
        "val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_200x25_Seed_6", "val"),
        "major_val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_200x25_Seed_6", "major_val"),
        "minor_val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_200x25_Seed_6", "minor_val")
    }
)

SAMPLED_IMAGENET_200X1000_50X100_SEED_6 = OrderedDict(
    {
        "num_classes": 250,
        "train_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_50x100_Seed_6", "train"),
        "val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_50x100_Seed_6", "val"),
        "major_val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_50x100_Seed_6", "major_val"),
        "minor_val_folder_path": os.path.join("./data/", "Sampled_ImageNet_200x1000_50x100_Seed_6", "minor_val")
    }
)

SAMPLED_IMAGENET = OrderedDict(
    {
        "num_classes": 250,
        "train_folder_path": os.path.join("./data/", "Sampled_ImageNet", "train"),
        "val_folder_path": os.path.join("./data/", "Sampled_ImageNet", "val"),
        "major_val_folder_path": os.path.join("./data/", "Sampled_ImageNet", "major_val"),
        "minor_val_folder_path": os.path.join("./data/", "Sampled_ImageNet", "minor_val")
    }
)

SAMPLED_IMAGENET_VAL = OrderedDict(
    {
        "num_classes": 250,
        "train_folder_path": os.path.join("./data/", "Sampled_ImageNet_Val"),
        "val_folder_path": os.path.join("./data/", "Sampled_ImageNet_Val")
    }
)

PROVIDED_DATA_FOLDERS = OrderedDict(
    {
        "Caltech-256": CALTECH_256,
        "Sampled_ImageNet_200x1000_200x25_Seed_6": SAMPLED_IMAGENET_200X1000_200X25_SEED_6,
        "Sampled_ImageNet_200x1000_50x100_Seed_6": SAMPLED_IMAGENET_200X1000_50X100_SEED_6,
        "Sampled_ImageNet": SAMPLED_IMAGENET,
        "Sampled_ImageNet_Val": SAMPLED_IMAGENET_VAL
    }
)
