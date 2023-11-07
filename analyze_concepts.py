import os
import shutil
import sys
import math
from tqdm import tqdm
from pprint import pprint
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from preprocess.imagenet_classes import IMAGENET2012_CLASSES_IDX
from modules.data_folders import PROVIDED_DATASETS
from modules.models import MODELS
from modules.models_exp import MODELS_EXP
from modules.utils import load_model, Recorder
from modules.losses import capped_lp_norm_hinge, orthogonality_l2_norm, PIController


##########################
# Basic settings
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

config = Recorder()

# dataset
# 'n02391049'  267, 'n02389026' 266.
config.dataset_name = "Sampled_ImageNet_800x500_200x0_Seed_6"
# 'n02391049'  340, 'n02389026' 339.
# config.dataset_name = "Sampled_ImageNet_500x1000_500x5_Seed_6"
# config.dataset_name = "Sampled_ImageNet_200x1000_200x25_Seed_6"  # 'n02391049' 134
# config.dataset_name = "Sampled_ImageNet"  # 'n02391049' 83
config.update(PROVIDED_DATASETS[config.dataset_name])
# model
config.use_model = "OriTextTopDownHierConceptPoolResNet"
config.backbone_name = "resnet50"
config.image_dim = 2048
config.image_spacial_dim = 7
config.text_embeds_path = "pre-trained/imagenet_zeroshot_simple_classifier.pt"
config.text_dim = 4096
config.detach_text_embeds = False
config.concept_dim = 512
config.num_low_concepts = 1024
config.norm_low_concepts = False
config.num_attended_low_concepts = 1024
config.num_high_concepts = 64
config.norm_high_concepts = False
config.num_attended_high_concepts = 64
config.low_high_max_function = "hardmax"
config.output_high_concepts_type = "original_high"
config.learnable_hierarchy = False
config.preset_hierarchy = True
config.detach_low_concepts = True
config.image_high_concept_num_heads = 4
config.image_high_concept_keep_head_dim = True
config.image_high_concept_max_function = "hard_gumbel"
config.image_high_concept_max_smoothing = 0.0
config.image_high_concept_threshold = None
config.patch_low_concept_num_heads = 1
config.patch_low_concept_keep_head_dim = True
config.patch_low_concept_max_function = "sparsemax"
config.patch_low_concept_max_smoothing = 0.0
config.patch_low_concept_threshold = None
config.image_patch_num_heads = 1
config.image_patch_keep_head_dim = True
config.image_patch_max_function = "softmax"
config.image_patch_max_smoothing = 0.0
config.contrastive_dim = 512
# loss
config.loss_low_sparsity_weight = 0.0
config.loss_low_sparsity_adaptive = False
config.loss_low_diversity_weight = 0.0
config.loss_high_sparsity_weight = 0.0
config.loss_high_sparsity_adaptive = False
config.loss_high_diversity_weight = 0.0
config.loss_aux_classification_weight = 1.0
# train
config.batch_size = 128
# device
config.dataloader_workers = 16
config.dataloader_pin_memory = True
# checkpoint
config.load_checkpoint_path = "./analyze_checkpoints/Sampled_ImageNet_800x500_200x0_Seed_6/OriTextTopDownHierConceptPoolResNet/202310241538_on_gpu_3/best_epoch_45_0.7087_0.2673_0.1864_49_62_4_38_4_16_143.3_9.1.pt"
config.checkpoint_desc = "resnet50_OriText_TopDown_4in64H_1024L"

# Confirm basic settings
print("\n"+"#"*100)
pprint(config.to_dict())
print("#"*100)

##########################
# Dataset and DataLoader
##########################

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=224,
                                     scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

# Load the dataset from the directory
train_dataset = ImageFolder(
    root=config.train_folder_path,
    transform=train_transform
)
train_idx_transform = dict()
for key, value in train_dataset.class_to_idx.items():
    train_idx_transform[value] = IMAGENET2012_CLASSES_IDX[key]
train_classes_idx = [
    train_idx_transform[idx] for idx in train_dataset.class_to_idx.values()
]
train_acc_mask_idx = [
    idx for idx in train_dataset.class_to_idx.values()
]

eval_dataset = ImageFolder(
    root=config.val_folder_path,
    transform=eval_transform
)
eval_idx_transform = dict()
for key, value in eval_dataset.class_to_idx.items():
    eval_idx_transform[value] = IMAGENET2012_CLASSES_IDX[key]
eval_classes_idx = [
    eval_idx_transform[idx] for idx in eval_dataset.class_to_idx.values()
]
eval_acc_mask_idx = [
    idx for idx in eval_dataset.class_to_idx.values()
]

tmp_major_dataset = ImageFolder(root=config.major_val_folder_path)
major_to_eval_idx_transform = dict()
for key, value in tmp_major_dataset.class_to_idx.items():
    major_to_eval_idx_transform[value] = eval_dataset.class_to_idx[key]


def major_to_eval_transform(target):
    return major_to_eval_idx_transform[target]


eval_major_dataset = ImageFolder(
    root=config.major_val_folder_path,
    transform=eval_transform,
    target_transform=major_to_eval_transform
)
eval_major_acc_mask_idx = [
    major_to_eval_transform(idx) for idx in eval_major_dataset.class_to_idx.values()
]

tmp_minor_dataset = ImageFolder(root=config.minor_val_folder_path)
minor_to_eval_idx_transform = dict()
for key, value in tmp_minor_dataset.class_to_idx.items():
    minor_to_eval_idx_transform[value] = eval_dataset.class_to_idx[key]


def minor_to_eval_transform(target):
    return minor_to_eval_idx_transform[target]


eval_minor_dataset = ImageFolder(
    root=config.minor_val_folder_path,
    transform=eval_transform,
    target_transform=minor_to_eval_transform
)
eval_minor_acc_mask_idx = [
    minor_to_eval_transform(idx) for idx in eval_minor_dataset.class_to_idx.values()
]


# Create DataLoader instances

train_loader = DataLoader(
    train_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_loader = DataLoader(
    eval_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_major_loader = DataLoader(
    eval_major_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)
eval_minor_loader = DataLoader(
    eval_minor_dataset, shuffle=False,
    batch_size=config.batch_size,
    num_workers=config.dataloader_workers,
    pin_memory=config.dataloader_pin_memory
)

##########################
# Model, loss and optimizer
##########################

PROVIDED_MODELS = OrderedDict(**MODELS, **MODELS_EXP)

model_parameters = dict(
    {
        "config": config
    }
)

model = PROVIDED_MODELS[config.use_model](**model_parameters).to(device)

load_model(model, config.load_checkpoint_path)

criterion = nn.CrossEntropyLoss()
low_sparsity_controller = PIController(
    kp=0.001, ki=0.00001,
    target_metric=config.num_attended_low_concepts,
    initial_weight=config.loss_low_sparsity_weight
)
high_sparsity_controller = PIController(
    kp=0.001, ki=0.00001,
    target_metric=config.num_attended_high_concepts,
    initial_weight=config.loss_high_sparsity_weight
)


def compute_loss(returned_dict, targets, train=False, metric_prefix="train_"):
    outputs = returned_dict["outputs"]
    aux_outputs = returned_dict.get("aux_outputs", None)
    # image_patch_attention_weight = returned_dict.get(
    #     "image_patch_attention_weight", None
    # )
    image_low_concept_attention_weight = returned_dict.get(
        "image_low_concept_attention_weight", None
    )
    image_high_concept_attention_weight = returned_dict.get(
        "image_high_concept_attention_weight", None
    )
    # patch_low_concept_attention_weight = returned_dict.get(
    #     "patch_low_concept_attention_weight", None
    # )
    # patch_high_concept_attention_weight = returned_dict.get(
    #     "patch_high_concept_attention_weight", None
    # )
    low_concept_cosine_similarity = returned_dict.get(
        "low_concept_cosine_similarity", None
    )
    high_concept_cosine_similarity = returned_dict.get(
        "high_concept_cosine_similarity", None
    )
    # low_high_hierarchy = returned_dict.get("low_high_hierarchy", None)

    def normalize_rows(input_tensor, epsilon=1e-10):
        input_tensor = input_tensor.to(torch.float)
        row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
        row_sums += epsilon
        normalized_tensor = input_tensor / row_sums
        return normalized_tensor

    if metric_prefix == "train_":
        num_classes = len(train_classes_idx)
    else:
        num_classes = len(eval_classes_idx)

    loss_cls_per_img = criterion(outputs, targets)  # B * K
    loss_img_per_cls = criterion(
        outputs.t(), normalize_rows(F.one_hot(targets, num_classes).t())
    )  # K * B
    loss_classification = (loss_cls_per_img + loss_img_per_cls) / 2.0

    if aux_outputs is None:
        loss_aux_classification = torch.tensor(0.0)
    else:
        loss_aux_cls_per_img = criterion(aux_outputs, targets)
        loss_aux_img_per_cls = criterion(
            aux_outputs.t(), normalize_rows(F.one_hot(targets, num_classes).t())
        )
        loss_aux_classification = (
            loss_aux_cls_per_img + loss_aux_img_per_cls
        ) / 2.0

    def num_attended_concepts_s99(attention_weights):
        with torch.no_grad():
            s99 = torch.quantile(
                torch.sum(
                    (attention_weights - 1e-7) > 0,
                    dim=1
                ).type(torch.float),
                0.99
            ).item()
        return s99

    if (image_low_concept_attention_weight is None) and (low_concept_cosine_similarity is None):
        loss_low_sparsity = torch.tensor(0.0)
        loss_low_diversity = torch.tensor(0.0)
    else:
        loss_low_sparsity = capped_lp_norm_hinge(
            image_low_concept_attention_weight,
            target=config.num_attended_low_concepts,
            gamma=1.0/config.num_attended_low_concepts,
            reduction="sum"
        )
        if config.loss_low_sparsity_adaptive and train:
            low_s99 = num_attended_concepts_s99(
                image_low_concept_attention_weight
            )
            config.loss_low_sparsity_weight = low_sparsity_controller.update(
                low_s99
            )
        loss_low_diversity = orthogonality_l2_norm(
            low_concept_cosine_similarity
        )

    if (image_high_concept_attention_weight is None) and (high_concept_cosine_similarity is None):
        loss_high_sparsity = torch.tensor(0.0)
        loss_high_diversity = torch.tensor(0.0)
    else:
        loss_high_sparsity = capped_lp_norm_hinge(
            image_high_concept_attention_weight,
            target=config.num_attended_high_concepts,
            gamma=1.0/config.num_attended_high_concepts,
            reduction="sum"
        )
        if config.loss_high_sparsity_adaptive and train:
            high_s99 = num_attended_concepts_s99(
                image_high_concept_attention_weight
            )
            config.loss_high_sparsity_weight = high_sparsity_controller.update(
                high_s99
            )
        loss_high_diversity = orthogonality_l2_norm(
            high_concept_cosine_similarity
        )

    loss = loss_classification + \
        config.loss_aux_classification_weight * loss_aux_classification + \
        config.loss_low_sparsity_weight * loss_low_sparsity + \
        config.loss_low_diversity_weight * loss_low_diversity + \
        config.loss_high_sparsity_weight * loss_high_sparsity + \
        config.loss_high_diversity_weight * loss_high_diversity

    return (
        loss,
        loss_classification,
        loss_aux_classification,
        loss_low_sparsity,
        loss_high_sparsity,
        loss_low_diversity,
        loss_high_diversity
    )


def custom_quantile(returned_dict: dict, key: str, dim: int, epsilon: float = 1e-7):
    """
    Compute the 10th, 50th, and 90th percentiles of a tensor along a specified dimension.

    Args:
        returned_dict: A dictionary containing the tensor to be processed.
        key: The key of the tensor to be processed.
        dim: The dimension along which to compute the percentiles.
        epsilon: A small value considered as zero.

    Returns:
        A tuple containing the 10th, 50th, and 90th percentiles of the tensor.
    """
    if returned_dict.get(key, None) is not None:
        count = torch.sum(
            (returned_dict.get(key).data - epsilon) > 0,
            dim=dim
        ).type(torch.float)
        s10 = torch.quantile(count, 0.10).item()
        s50 = torch.quantile(count, 0.50).item()
        s90 = torch.quantile(count, 0.90).item()
    else:
        s10 = -1
        s50 = -1
        s90 = -1
    return s10, s50, s90


##########################
# Analyze pipeline
##########################


def run_epoch(desc, model, dataloader, acc_mask_idx, metric_prefix=""):

    model.eval()

    metric_dict = dict()
    screen_dict = dict()

    label = []
    patch_low_concept_attention_weight = []
    patch_high_concept_attention_weight = []
    image_low_concept_attention_weight = []
    image_high_concept_attention_weight = []
    low_high_hierarchy = None

    step = 0
    with tqdm(
        total=len(dataloader),
        desc=desc,
        postfix=dict,
        mininterval=1,
        file=sys.stdout,
        dynamic_ncols=True
    ) as pbar:
        for data, targets in dataloader:
            # data process
            data = data.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                if metric_prefix == "train_":
                    returned_dict = model(data, train_classes_idx)
                else:
                    returned_dict = model(data, eval_classes_idx)
                (
                    loss,
                    loss_classification,
                    loss_aux_classification,
                    loss_low_sparsity,
                    loss_high_sparsity,
                    loss_low_diversity,
                    loss_high_diversity
                ) = compute_loss(
                    returned_dict, targets, train=False, metric_prefix=metric_prefix
                )

            if returned_dict.get("image_high_concept_attention_weight", None) is not None:
                label.append(targets.cpu().numpy())
                # patch_low_concept_attention_weight.append(
                #     returned_dict["patch_low_concept_attention_weight"].detach(
                #     ).cpu().numpy()
                # )
                patch_high_concept_attention_weight.append(
                    returned_dict["patch_high_concept_attention_weight"].detach(
                    ).cpu().numpy()
                )
                image_low_concept_attention_weight.append(
                    returned_dict["image_low_concept_attention_weight"].detach(
                    ).cpu().numpy()
                )
                image_high_concept_attention_weight.append(
                    returned_dict["image_high_concept_attention_weight"].detach(
                    ).cpu().numpy()
                )
                low_high_hierarchy = returned_dict["low_high_hierarchy"].detach(
                ).cpu().numpy()

            # display the metrics
            with torch.no_grad():

                acc = (torch.argmax(returned_dict["outputs"].data,
                                    1) == targets).sum() / targets.size(0)

                mask = torch.zeros_like(returned_dict["outputs"].data)
                mask[:, acc_mask_idx] = 1
                acc_subset = (torch.argmax(returned_dict["outputs"].data * mask,
                                           1) == targets).sum() / targets.size(0)

                if returned_dict.get("aux_outputs", None) is not None:
                    aux_acc = (torch.argmax(returned_dict["aux_outputs"].data,
                                            1) == targets).sum() / targets.size(0)
                    aux_acc_subset = (torch.argmax(returned_dict["aux_outputs"].data * mask,
                                                   1) == targets).sum() / targets.size(0)
                else:
                    aux_acc = torch.tensor(0.0)
                    aux_acc_subset = torch.tensor(0.0)

                pfi_s10, pfi_s50, pfi_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="image_patch_attention_weight",
                    dim=1
                )

                lcfi_s10, lcfi_s50, lcfi_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="image_low_concept_attention_weight",
                    dim=1
                )

                hcfi_s10, hcfi_s50, hcfi_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="image_high_concept_attention_weight",
                    dim=1
                )

                lcfp_s10, lcfp_s50, lcfp_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="patch_low_concept_attention_weight",
                    dim=2
                )

                hcfp_s10, hcfp_s50, hcfp_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="patch_high_concept_attention_weight",
                    dim=2
                )

                lfh_s10, lfh_s50, lfh_s90 = custom_quantile(
                    returned_dict=returned_dict,
                    key="low_high_hierarchy",
                    dim=0
                )

            def update_metric_dict(key, value, average=True, verbose=True):
                if average:
                    metric_dict[metric_prefix + key] = (
                        metric_dict.get(
                            metric_prefix + key, 0
                        ) * step + value
                    ) / (step + 1)
                    if verbose:
                        screen_dict[key] = (
                            screen_dict.get(key, 0) * step + value
                        ) / (step + 1)
                else:
                    metric_dict[metric_prefix + key] = value
                    if verbose:
                        screen_dict[key] = value

            update_metric_dict("A", acc.item())
            update_metric_dict("A_sub", acc_subset.item())
            update_metric_dict("A_aux", aux_acc.item())
            update_metric_dict("A_auxsub", aux_acc_subset.item())
            update_metric_dict("L", loss.item())
            update_metric_dict("L_cls", loss_classification.item())
            update_metric_dict("L_aux", loss_aux_classification.item())
            update_metric_dict("L_lsps", loss_low_sparsity.item())
            update_metric_dict("L_hsps", loss_high_sparsity.item())
            update_metric_dict("L_ldvs", loss_low_diversity.item())
            update_metric_dict("L_hdvs", loss_high_diversity.item())
            update_metric_dict(
                "W_lsps", config.loss_low_sparsity_weight, average=False
            )
            update_metric_dict(
                "W_hsps", config.loss_high_sparsity_weight, average=False
            )
            update_metric_dict("pfi_s10", pfi_s10, verbose=False)
            update_metric_dict("pfi_s50", pfi_s50, verbose=False)
            update_metric_dict("pfi_s90", pfi_s90)
            update_metric_dict("lcfi_s10", lcfi_s10, verbose=False)
            update_metric_dict("lcfi_s50", lcfi_s50, verbose=False)
            update_metric_dict("lcfi_s90", lcfi_s90)
            update_metric_dict("hcfi_s10", hcfi_s10, verbose=False)
            update_metric_dict("hcfi_s50", hcfi_s50, verbose=False)
            update_metric_dict("hcfi_s90", hcfi_s90)
            update_metric_dict("lcfp_s10", lcfp_s10, verbose=False)
            update_metric_dict("lcfp_s50", lcfp_s50, verbose=False)
            update_metric_dict("lcfp_s90", lcfp_s90)
            update_metric_dict("hcfp_s10", hcfp_s10, verbose=False)
            update_metric_dict("hcfp_s50", hcfp_s50, verbose=False)
            update_metric_dict("hcfp_s90", hcfp_s90)
            update_metric_dict("lfh_s10", lfh_s10, verbose=False)
            update_metric_dict("lfh_s50", lfh_s50, verbose=False)
            update_metric_dict("lfh_s90", lfh_s90)

            pbar.set_postfix(screen_dict)
            pbar.update(1)

            step += 1
    return (
        label,
        patch_low_concept_attention_weight,
        patch_high_concept_attention_weight,
        image_low_concept_attention_weight,
        image_high_concept_attention_weight,
        low_high_hierarchy
    )


desc = f"Training"
(
    train_label,
    train_patch_low_concept_attention_weight,
    train_patch_high_concept_attention_weight,
    train_image_low_concept_attention_weight,
    train_image_high_concept_attention_weight,
    train_low_high_hierarchy
) = run_epoch(
    desc, model, train_loader, train_acc_mask_idx, metric_prefix="train_"
)

desc = f"Evaluate"
(
    eval_label,
    eval_patch_low_concept_attention_weight,
    eval_patch_high_concept_attention_weight,
    eval_image_low_concept_attention_weight,
    eval_image_high_concept_attention_weight,
    eval_low_high_hierarchy
) = run_epoch(
    desc, model, eval_loader, eval_acc_mask_idx, metric_prefix="val_"
)

desc = f"MajorVal"
(
    eval_major_label,
    eval_major_patch_low_concept_attention_weight,
    eval_major_patch_high_concept_attention_weight,
    eval_major_image_low_concept_attention_weight,
    eval_major_image_high_concept_attention_weight,
    eval_major_low_high_hierarchy
) = run_epoch(
    desc, model, eval_major_loader, eval_major_acc_mask_idx, metric_prefix="major_"
)

desc = f"MinorVal"
(
    eval_minor_label,
    eval_minor_patch_low_concept_attention_weight,
    eval_minor_patch_high_concept_attention_weight,
    eval_minor_image_low_concept_attention_weight,
    eval_minor_image_high_concept_attention_weight,
    eval_minor_low_high_hierarchy
) = run_epoch(
    desc, model, eval_minor_loader, eval_minor_acc_mask_idx, metric_prefix="minor_"
)


analyze_label = np.concatenate(train_label, axis=0)
# analyze_patch_low_concept_attention_weight = np.concatenate(
#     train_patch_low_concept_attention_weight, axis=0
# )
analyze_patch_high_concept_attention_weight = np.concatenate(
    train_patch_high_concept_attention_weight, axis=0
)
analyze_image_low_concept_attention_weight = np.concatenate(
    train_image_low_concept_attention_weight, axis=0
)
analyze_image_high_concept_attention_weight = np.concatenate(
    train_image_high_concept_attention_weight, axis=0
)
visual_image_low_concept_attention_weight = np.concatenate(
    eval_major_image_low_concept_attention_weight, axis=0
)
visual_image_high_concept_attention_weight = np.concatenate(
    eval_major_image_high_concept_attention_weight, axis=0
)
analyze_low_high_hierarchy = train_low_high_hierarchy

analyze_label2name_dict = {
    value: key for key, value in train_dataset.class_to_idx.items()
}

image_high_concept_attention_weight_per_label = dict()
for label, attention in zip(analyze_label, analyze_image_high_concept_attention_weight):
    if label not in image_high_concept_attention_weight_per_label.keys():
        image_high_concept_attention_weight_per_label[label] = []
    image_high_concept_attention_weight_per_label[label].append(attention)
label_high_concept_attention_weight = dict()
for label, attention_list in image_high_concept_attention_weight_per_label.items():
    assert label not in label_high_concept_attention_weight.keys()
    label_high_concept_attention_weight[label] = np.mean(
        attention_list, axis=0)

image_low_concept_attention_weight_per_label = dict()
for label, attention in zip(analyze_label, analyze_image_low_concept_attention_weight):
    if label not in image_low_concept_attention_weight_per_label.keys():
        image_low_concept_attention_weight_per_label[label] = []
    image_low_concept_attention_weight_per_label[label].append(attention)
label_low_concept_attention_weight = dict()
for label, attention_list in image_low_concept_attention_weight_per_label.items():
    assert label not in label_low_concept_attention_weight.keys()
    label_low_concept_attention_weight[label] = np.mean(attention_list, axis=0)


# def summarize_label_concept(label_concept_attention_weight, threshold):
#     label_concept_dict = dict()
#     for label, attention_weight in label_concept_attention_weight.items():
#         indices = np.where(attention_weight >= threshold)
#         values = attention_weight[indices]
#         concept_dict = dict(zip(indices[0], values))
#         label_concept_dict[label] = dict(
#             sorted(concept_dict.items(), key=lambda x: x[1], reverse=True)
#         )
#     return dict(sorted(label_concept_dict.items(), key=lambda x: x[0]))


def summarize_label_concept(label_concept_attention_weight, k):
    label_concept_dict = dict()
    for label, attention_weight in label_concept_attention_weight.items():
        indices = np.argpartition(attention_weight, -k)[-k:]
        values = attention_weight[indices]
        concept_dict = dict(zip(indices, values))
        label_concept_dict[label] = dict(
            sorted(concept_dict.items(), key=lambda x: x[1], reverse=True)
        )
    return dict(sorted(label_concept_dict.items(), key=lambda x: x[0]))


label_high_concept_dict = summarize_label_concept(
    label_high_concept_attention_weight, 4
)
label_low_concept_dict = summarize_label_concept(
    label_low_concept_attention_weight, 64
)


def summary_concept_label(label_concept_dict):
    concept_label_dict = dict()
    for label, concept_dict in label_concept_dict.items():
        for concept, weight in concept_dict.items():
            if concept not in concept_label_dict.keys():
                concept_label_dict[concept] = dict()
            concept_label_dict[concept][label] = weight
    for concept, label_dict in concept_label_dict.items():
        concept_label_dict[concept] = dict(
            sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
        )
    return dict(sorted(concept_label_dict.items(), key=lambda x: x[0]))


high_concept_label_dict = summary_concept_label(label_high_concept_dict)
low_concept_label_dict = summary_concept_label(label_low_concept_dict)


def summary_high_low(low_high_hierarchy):
    high_low_dict = dict()
    rows, cols = np.where(low_high_hierarchy > 0)
    for col, row in zip(cols, rows):
        if col not in high_low_dict.keys():
            high_low_dict[col] = []
        high_low_dict[col].append(row)
    return dict(sorted(high_low_dict.items(), key=lambda x: x[0]))


high_low_dict = summary_high_low(analyze_low_high_hierarchy)

# visual_images = [sample[0] for sample in train_dataset.samples]
visual_images = [sample[0] for sample in eval_major_dataset.samples]


def get_top_k_filenames(filenames, arr, k):
    # 获取数组每列的前 k 个最大值的索引
    top_k_indices = np.argsort(-arr, axis=0)[:k]

    # 使用索引从文件名列表中获取对应的文件名
    result = []
    for col in range(top_k_indices.shape[1]):
        col_top_k_filenames = [filenames[i] for i in top_k_indices[:, col]]
        result.append(col_top_k_filenames)

    return result


m = 10
high_concept_images = get_top_k_filenames(
    visual_images, visual_image_high_concept_attention_weight, m * m
)
low_concept_images = get_top_k_filenames(
    visual_images, visual_image_low_concept_attention_weight, m * m
)


def stitch_images(image_paths, m):
    # 获取所有图片的大小
    sizes = [Image.open(p).size for p in image_paths]

    # 计算平均长宽比
    avg_ratio = sum(width / height for width, height in sizes) / len(sizes)

    # 计算目标宽度和高度
    target_height = min(height for width, height in sizes)
    target_width = int(target_height * avg_ratio)

    # 调整每张图片的大小
    images = [Image.open(p) for p in image_paths]
    resized_images = [img.resize(
        (target_width, target_height), Image.LANCZOS
    ) for img in images]

    # 计算大图的大小
    n = math.ceil(len(image_paths) / m)
    stitched_image = Image.new('RGB', (m * target_width, n * target_height))

    # 遍历图片，将每个图片粘贴到大图上
    for index, img in enumerate(resized_images):
        # 计算当前图片在大图中的位置
        row = index // m
        col = index % m

        # 将图片粘贴到大图上
        stitched_image.paste(img, (col * target_width, row * target_height))

    return stitched_image


def copy_files_to_folder(file_list, target_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历文件列表，将每个文件复制到目标文件夹
    for file_path in file_list:
        if os.path.isfile(file_path):
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 构建目标文件路径
            target_path = os.path.join(target_folder, file_name)
            # 复制文件
            shutil.copy(file_path, target_path)
        else:
            print(f"Warning: {file_path} is not a valid file.")


for i, image_paths in enumerate(high_concept_images):
    if i not in high_low_dict.keys():
        continue
    output_dir = os.path.join(
        os.path.dirname(config.load_checkpoint_path),
        config.checkpoint_desc,
        "high_concept_images"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_image = stitch_images(image_paths, m)
    result_image.save(os.path.join(output_dir, f"high_concept_{i}.jpg"))
    copy_files_to_folder(image_paths, os.path.join(
        output_dir, f"high_concept_{i}_top_images"))

for i, image_paths in enumerate(low_concept_images):
    output_dir = os.path.join(
        os.path.dirname(config.load_checkpoint_path),
        config.checkpoint_desc,
        "low_concept_images"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_image = stitch_images(image_paths, m)
    result_image.save(os.path.join(output_dir, f"low_concept_{i}.jpg"))
    copy_files_to_folder(image_paths, os.path.join(
        output_dir, f"low_concept_{i}_top_images"))
