import torch
import torch.nn as nn
import torchvision.models as models
from utils import Recorder
from typing import Callable, Dict, List, Tuple, Any


def get_callable_backbone(backbone_name: str) -> Callable:
    """
    Returns a callable function for the specified backbone model.

    Args:
        backbone_name (str): The name of the backbone model.

    Returns:
        Callable: A callable function for the specified backbone model.

    Raises:
        ValueError: If the provided backbone_name is not valid.
    """
    callable_backbones = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50
        # Add other models here
    }

    if backbone_name in callable_backbones:
        return callable_backbones[backbone_name]
    else:
        raise ValueError(f"Invalid backbone name: {backbone_name}")


class ResNet(nn.Module):
    """
    ResNet class, a custom implementation of ResNet architecture using a configurable backbone.

    Attributes:
        backbone_name (str): The name of the backbone model.
        num_classes (int): The number of output classes.
        backbone_callable (Callable): A callable function for the specified backbone model.
        backbone (nn.Module): The backbone model instance.
    """

    def __init__(self, config: Recorder):
        super(ResNet, self).__init__()

        self.parse_config(config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=self.num_classes
        )

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.num_classes = config.get("num_classes")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"outputs": self.backbone(x)}


class ContrastImageTextEmbeds(nn.Module):
    """
    A PyTorch module for contrastive learning of image and text embeddings.

    This module applies layer normalization and linear projection to both image
    and text embeddings, followed by L2 normalization. The output is the dot
    product of the image and text embeddings, scaled by an exponential scaling factor.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
    """

    def __init__(self, embed_dim: int):
        super(ContrastImageTextEmbeds, self).__init__()

        # Layer normalization for image and text embeddings
        self.image_post_layernorm = nn.LayerNorm(embed_dim)
        self.text_post_layernorm = nn.LayerNorm(embed_dim)

        # Linear projection layers for image and text embeddings
        self.image_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Scaling factor for the output
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ContrastImageTextEmbeds module.

        Args:
            image_embeds (torch.Tensor): Image embeddings of shape (batch_size, embed_dim).
            text_embeds (torch.Tensor): Text embeddings of shape (batch_size, embed_dim).

        Returns:
            torch.Tensor: The dot product of image and text embeddings, scaled by the scaling factor.
        """
        # Apply layer normalization and linear projection to image embeddings
        image_embeds = self.image_projection(
            self.image_post_layernorm(image_embeds)
        )
        # Normalize image embeddings using L2 norm
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        # Apply layer normalization and linear projection to text embeddings
        text_embeds = self.text_projection(
            self.text_post_layernorm(text_embeds)
        )
        # Normalize text embeddings using L2 norm
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # Calculate the exponential of the scaling factor
        logit_scale_exp = self.logit_scale.exp()
        # Compute the dot product of image and text embeddings, scaled by the scaling factor
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale_exp

        return outputs


class TextEncoderSimulator(nn.Module):
    """  
    TextEncoderSimulator simulates a text encoder, directly returning text embeddings.  
    """

    def __init__(self, text_embeds_path: str):
        """  
        Initialize TextEncoderSimulator.  

        :param text_embeds_path: Path to the text embedding tensor  
        """
        super(TextEncoderSimulator, self).__init__()
        self.text_embeds = nn.Parameter(
            torch.load(text_embeds_path).t(),
            requires_grad=False
        )

    def forward(self, idx: List) -> torch.Tensor:
        """  
        Return the corresponding text embeddings based on row indices.  

        :param idx: List of row indices to retrieve the embeddings for  
        :return: The corresponding text embeddings  
        """
        return self.text_embeds[idx, :]


class OriTextResNet(nn.Module):
    def __init__(self, config: Recorder):
        super(OriTextResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=self.text_embeds.size(0)
        )

        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-1]
        )
        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path)

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.text_embeds_path = config.get("text_embeds_path")
        self.contrastive_dim = config.get("contrastive_dim")

    def transform_embed_dim(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform embedding dimensions if necessary.

        Args:
            image_embeds (torch.Tensor): Image embeddings.
            text_embeds (torch.Tensor): Text embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and text embeddings.
        """
        if image_embeds.size(1) != self.contrastive_dim:
            image_dim_transformer = nn.Linear(
                image_embeds.size(1), self.contrastive_dim
            )
            image_embeds = image_dim_transformer(image_embeds)
        if text_embeds.size(1) != self.contrastive_dim:
            text_dim_transformer = nn.Linear(
                text_embeds.size(1), self.contrastive_dim
            )
            text_embeds = text_dim_transformer(text_embeds)
        return image_embeds, text_embeds

    def forward(self, x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor, typically images.
            classes_idx (List): List of class indices.

        Returns:
            Dict[str, torch.Tensor]: Model outputs.
        """
        image_embeds = torch.flatten(
            self.image_encoder(x), start_dim=1
        )
        text_embeds = self.text_encoder(classes_idx)

        image_embeds, text_embeds = self.transform_embed_dim(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        outputs = self.contrast(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        return {"outputs": outputs}


class Concepts(nn.Module):
    def __init__(self, num_concepts: int, concepts_dim: int):
        super(Concepts, self).__init__()

        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, concepts_dim)
        )

        nn.init.xavier_uniform_(self.concepts)

    def forward(self, norm_concepts: bool) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Concepts module.

        Args:
            norm_concepts (bool): Whether to normalize concept vectors.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing concepts and their cosine similarity matrix.
        """
        # Normalize concept vectors if requested.
        normalized_concepts = self.concepts / \
            self.concepts.norm(dim=1, keepdim=True)

        # Compute the cosine similarity matrix between normalized concepts.
        concept_cosine_similarity = torch.matmul(
            normalized_concepts, normalized_concepts.t()
        )

        return {
            "concepts": normalized_concepts if norm_concepts else self.concepts,
            "concept_cosine_similarity": concept_cosine_similarity
        }
