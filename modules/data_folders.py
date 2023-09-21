import os
from collections import OrderedDict
from typing import List


def create_path(base_folder: str, split: str, subfolders: bool) -> str:
    """
    Create a folder path based on the base folder, split, and subfolders flag.

    Args:
        base_folder (str): The name of the base folder containing the dataset.
        split (str): The dataset split (e.g., "train" or "val").
        subfolders (bool): Whether the dataset is organized in subfolders.

    Returns:
        str: The folder path.
    """
    if subfolders:
        return os.path.join("./data/", base_folder, split)
    else:
        return os.path.join("./data/", base_folder)


def create_dataset_dict(
    num_classes: int,
    base_folder: str,
    splits: List[str],
    subfolders: bool = True,
) -> OrderedDict:
    """
    Create a dataset dictionary with folder paths and number of classes.

    Args:
        num_classes (int): The number of classes in the dataset.
        base_folder (str): The name of the base folder containing the dataset.
        splits (List[str]): A list of dataset splits (e.g., ["train", "val"]).
        subfolders (bool, optional): Whether the dataset is organized in subfolders. Defaults to True.

    Returns:
        OrderedDict: An ordered dictionary with the number of classes and folder paths.
    """
    return OrderedDict(
        {
            "num_classes": num_classes,
            **{
                f"{split}_folder_path": create_path(base_folder, split, subfolders)
                for split in splits
            }
        }
    )


# Create dataset dictionaries for different datasets
CALTECH_256 = create_dataset_dict(
    num_classes=256,
    base_folder="Caltech-256",
    splits=["train", "val"],
    subfolders=False
)
SAMPLED_IMAGENET_500X1000_200X5_SEED_6 = create_dataset_dict(
    num_classes=700,
    base_folder="Sampled_ImageNet_500x1000_200x5_Seed_6",
    splits=["train", "val", "major_val", "minor_val"]
)
SAMPLED_IMAGENET_500X1000_500X5_SEED_6 = create_dataset_dict(
    num_classes=1000,
    base_folder="Sampled_ImageNet_500x1000_500x5_Seed_6",
    splits=["train", "val", "major_val", "minor_val"]
)
SAMPLED_IMAGENET_500X1000_200X0_SEED_6 = create_dataset_dict(
    num_classes=700,
    base_folder="Sampled_ImageNet_500x1000_200x0_Seed_6",
    splits=["train", "val", "major_val", "minor_val"]
)
SAMPLED_IMAGENET_500X1000_200X0_SEED_8 = create_dataset_dict(
    num_classes=700,
    base_folder="Sampled_ImageNet_500x1000_200x0_Seed_8",
    splits=["train", "val", "major_val", "minor_val"]
)
SAMPLED_IMAGENET_800X500_200X0_SEED_6 = create_dataset_dict(
    num_classes=1000,
    base_folder="Sampled_ImageNet_800x500_200x0_Seed_6",
    splits=["train", "val", "major_val", "minor_val"]
)
SAMPLED_IMAGENET_800X500_200X0_SEED_8 = create_dataset_dict(
    num_classes=1000,
    base_folder="Sampled_ImageNet_800x500_200x0_Seed_8",
    splits=["train", "val", "major_val", "minor_val"]
)


# Create an ordered dictionary of provided datasets
PROVIDED_DATASETS = OrderedDict(
    {
        "Caltech-256": CALTECH_256,
        "Sampled_ImageNet_500x1000_200x5_Seed_6": SAMPLED_IMAGENET_500X1000_200X5_SEED_6,
        "Sampled_ImageNet_500x1000_500x5_Seed_6": SAMPLED_IMAGENET_500X1000_500X5_SEED_6,
        "Sampled_ImageNet_500x1000_200x0_Seed_6": SAMPLED_IMAGENET_500X1000_200X0_SEED_6,
        "Sampled_ImageNet_500x1000_200x0_Seed_8": SAMPLED_IMAGENET_500X1000_200X0_SEED_8,
        "Sampled_ImageNet_800x500_200x0_Seed_6": SAMPLED_IMAGENET_800X500_200X0_SEED_6,
        "Sampled_ImageNet_800x500_200x0_Seed_8": SAMPLED_IMAGENET_800X500_200X0_SEED_8,
    }
)
