import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sparsemax import Sparsemax
from .utils import Recorder
from typing import Callable, Dict, List, Tuple, Any
from collections import OrderedDict


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


def get_max_function(max_function_name: str, dim: int = -1) -> Any:
    """
    Returns a PyTorch module for performing max operation along a specified dimension.

    Args:
        max_function_name (str): The name of the max function to use. Valid values are "softmax", "sparsemax", "gumbel", and "hard_gumbel".
        dim (int, optional): The dimension along which to apply the max function. Default: -1.

    Returns:
        A PyTorch module for performing max operation along the specified dimension.

    Raises:
        ValueError: If `max_function_name` is not one of the valid values.

    Examples:
        >>> max_fn = get_max_function("softmax", 1)
        >>> x = torch.randn(2, 3)
        >>> y = max_fn(x)
    """
    max_functions = {
        "softmax": nn.Softmax(dim=dim),
        "sparsemax": Sparsemax(dim=dim),
        "gumbel": GumbelSoftmax(dim=dim),
        "hard_gumbel": GumbelSoftmax(hard=True, dim=dim)
    }

    if max_function_name in max_functions:
        return max_functions[max_function_name]
    else:
        raise ValueError(f"Invalid max function: {max_function_name}")


class GumbelSoftmax(nn.Module):
    """
    Gumbel Softmax module.

    Args:
        hard (bool, optional): Whether to use hard Gumbel softmax (default: False).
        dim (int, optional): The dimension along which the Gumbel softmax is applied (default: -1).
        eps (float, optional): A small value added to the denominator for numerical stability (default: 1e-10).

    Attributes:
        temperature (torch.Tensor): A learnable parameter that controls the temperature of the Gumbel-Softmax function.
        tau (float): A constant that controls the behavior of the Gumbel-Softmax function.
        hard (bool): If True, uses hard Gumbel-Softmax for discrete sampling.
        dim (int): The dimension along which to apply the Gumbel-Softmax function.
        eps (float): A small value to ensure numerical stability.

    Examples:
        >>> gs = GumbelSoftmax()
        >>> x = torch.randn(2, 10)
        >>> y = gs(x)
    """

    def __init__(self, hard=False, dim=-1, eps=1e-10):
        super(GumbelSoftmax, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(0.0))
        self.tau = 1.0
        self.hard = hard
        self.dim = dim
        self.eps = eps

    def forward(self, logits):
        if self.training:
            return F.gumbel_softmax(logits * self.temperature.exp(), tau=self.tau, hard=self.hard, dim=self.dim, eps=self.eps)
        else:
            index = logits.max(self.dim, keepdim=True)[1]
            logits_hard = torch.zeros_like(
                logits,
                memory_format=torch.legacy_contiguous_format
            ).scatter_(self.dim, index, 1.0)
        return logits_hard


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
            weights=None, num_classes=0
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-1]
        )
        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.text_embeds_path = config.get("text_embeds_path")
        self.contrastive_dim = config.get("contrastive_dim")

    def transform_embed_dim(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Transform the dimension of the input tensor from `embeds.size(1)` to `self.contrastive_dim`.

        Args:
            embeds (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.

        If the dimension of the input tensor is already equal to `self.contrastive_dim`, the input tensor is returned directly.
        Otherwise, a `nn.Linear` module is created to transform the dimension of the input tensor to `self.contrastive_dim`,
        and the input tensor is passed to the module for transformation. The transformed tensor is then returned.

        """
        if embeds.size(1) != self.contrastive_dim:
            embeds_dim_transformer = nn.Linear(
                embeds.size(1), self.contrastive_dim
            )
            embeds = embeds_dim_transformer(embeds)
        return embeds

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
        image_embeds = self.transform_embed_dim(embeds=image_embeds)

        text_embeds = self.text_encoder(classes_idx)
        text_embeds = self.transform_embed_dim(embeds=text_embeds)

        outputs = self.contrast(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        return {"outputs": outputs}


class Concepts(nn.Module):
    def __init__(self, num_concepts: int, concept_dim: int):
        """
        Initializes a new instance of the `Concepts` class.

        Args:
            num_concepts (int): The number of concepts in the concept matrix.
            concept_dim (int): The dimension of each concept vector.
        """
        super(Concepts, self).__init__()

        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, concept_dim)
        )

        nn.init.xavier_uniform_(self.concepts)

    def forward(self, norm_concepts: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the `Concepts` module.

        Args:
            norm_concepts (bool): Whether to normalize the concept vectors or not.

        Returns:
            A tuple containing the concept matrix and the cosine similarity matrix between the normalized
            concept vectors.
        """
        # Normalize concept vectors if requested.
        normalized_concepts = self.concepts / \
            self.concepts.norm(dim=1, keepdim=True)
        returned_concepts = normalized_concepts if norm_concepts else self.concepts

        # Compute the cosine similarity matrix between normalized concepts.
        concept_cosine_similarity = torch.matmul(
            normalized_concepts, normalized_concepts.t()
        )

        return returned_concepts, concept_cosine_similarity


class ModifiedMultiHeadAttention(nn.Module):
    """
    A modified multi-head attention module in PyTorch.

    Args:
        query_dim (int): The dimension of the query vector.
        key_dim (int): The dimension of the key vector.
        n_head (int): The number of attention heads.
        keep_head_dim (bool): Whether to keep the head dimension or not.
        max_function_name (str): The name of the function used to calculate the maximum value in the attention matrix.
        max_smoothing (float, optional): The amount of smoothing applied to the maximum value. Defaults to 0.0.

    Attributes:
        n_head (int): The number of attention heads.
        max_smoothing (float): The amount of smoothing applied to the maximum value.
        d_head (int): The dimension of each attention head.
        q_linear (nn.Linear): The linear layer for the query vector.
        k_linear (nn.Linear): The linear layer for the key vector.
        max_function (function): The function used to calculate the maximum value in the attention matrix.

    Methods:
        smooth_max(max_output: torch.Tensor) -> torch.Tensor:
            Smooths the maximum value in the attention matrix.
        forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the output of the multi-head attention module.

    """

    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 n_head: int,
                 keep_head_dim: bool,
                 max_function_name: str,
                 max_smoothing: float = 0.0) -> None:
        super(ModifiedMultiHeadAttention, self).__init__()

        self.n_head = n_head
        assert max_smoothing >= 0.0 and max_smoothing <= 1.0
        self.max_smoothing = max_smoothing

        if keep_head_dim:
            self.d_head = key_dim
            self.q_linear = nn.Linear(query_dim, key_dim*n_head)
            self.k_linear = nn.Linear(key_dim, key_dim*n_head)
        else:
            assert key_dim % n_head == 0
            self.d_head = key_dim // n_head
            self.q_linear = nn.Linear(query_dim, key_dim)
            self.k_linear = nn.Linear(key_dim, key_dim)

        self.max_function = get_max_function(
            max_function_name=max_function_name,
            dim=-1
        )

    def smooth_max(self, max_output: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.max_smoothing > 0.0 and self.training:
            max_output = max_output * (1.0 - self.max_smoothing) + \
                self.max_smoothing / max_output.size(dim)
        return max_output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the output of the multi-head attention module.

        Args:
            q (torch.Tensor): The query vector. [B_q, L_q, D_q]
            k (torch.Tensor): The key vector. [B_kv, L_kv, D_k]
            v (torch.Tensor): The value vector. [B_kv, L_kv, D_v]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor, attention tensor.

        """
        q = self.q_linear(q).view(
            q.size(0), -1, self.n_head, self.d_head
        ).transpose(1, 2)  # [B_q, H, L_q, d_head]
        k = self.k_linear(k).view(
            k.size(0), -1, self.n_head, self.d_head
        ).transpose(1, 2)  # [B_kv, H, L_kv, d_head]

        attn = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.d_head ** 0.5)  # [B_q, H, L_q, L_kv]
        attn = self.max_function(attn)  # [B_q, H, L_q, L_kv]

        attn = attn.mean(dim=1)  # [B_q, L_q, L_kv]
        output = torch.matmul(self.smooth_max(attn), v)  # [B_q, L_q, D_v]

        return output, attn


class Conceptualizer(nn.Module):
    """
    A PyTorch module that computes a conceptual vector representation of input data using attention mechanism.

    Args:
        feature_dim (int): The dimension of the input feature vector.
        concept_dim (int): The dimension of the concept vector.
        n_head (int): The number of attention heads.
        keep_head_dim (bool): Whether to keep the dimension of each attention head or not.
        max_function_name (str): The name of the maximum function to use for attention calculation.
        max_smoothing (float, optional): The smoothing factor for the maximum function. Defaults to 0.0.

    Inputs:
        x (torch.Tensor): The input feature tensor of shape (batch_size, feature_dim).
        concepts (torch.Tensor): The concept tensor of shape (num_concepts, concept_dim).

    Outputs:
        A tuple containing:
        - conceptual_x (torch.Tensor): The conceptual vector representation of the input tensor of shape (batch_size, concept_dim).
        - concept_attention_weight (torch.Tensor): The attention weight of the concept tensor of shape (batch_size, num_concepts).
    """

    def __init__(self,
                 feature_dim: int,
                 concept_dim: int,
                 n_head: int,
                 keep_head_dim: bool,
                 max_function_name: str,
                 max_smoothing: float = 0.0):
        super(Conceptualizer, self).__init__()

        self.conceptual_attention = ModifiedMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=concept_dim,
            n_head=n_head,
            keep_head_dim=keep_head_dim,
            max_function_name=max_function_name,
            max_smoothing=max_smoothing
        )

    def forward(self, x: torch.Tensor, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B, 1, D_q]
        concepts = concepts.unsqueeze(0)  # [1, N, D_kv]
        conceptual_x, concept_attention_weight = self.conceptual_attention(
            x, concepts, concepts
        )  # [B, 1, D_kv], [B, 1, N]

        # [B, D_kv], [B, N]
        return conceptual_x.squeeze(1), concept_attention_weight.squeeze(1)


class OriTextConceptualResNet(nn.Module):
    def __init__(self, config: Recorder):
        super(OriTextConceptualResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=0
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-1]
        )
        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.concepts = Concepts(
            num_concepts=self.num_concepts,
            concept_dim=self.concept_dim
        )
        self.conceptualizer = Conceptualizer(
            feature_dim=self.image_dim,
            concept_dim=self.concept_dim,
            n_head=self.n_head,
            keep_head_dim=self.keep_head_dim,
            max_function_name=self.max_function_name,
            max_smoothing=self.max_smoothing
        )

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)

    def parse_config(self, config: Recorder) -> None:

        self.backbone_name = config.get("backbone_name")
        self.image_dim = config.get("image_dim")

        self.text_embeds_path = config.get("text_embeds_path")

        self.concept_dim = config.get("concept_dim")
        self.num_concepts = config.get("num_low_concepts")
        self.norm_concepts = config.get("norm_low_concepts")

        self.n_head = config.get("image_low_concept_num_heads")
        self.keep_head_dim = config.get("image_low_concept_keep_head_dim")
        self.max_function_name = config.get("image_low_concept_max_function")
        self.max_smoothing = config.get("image_low_concept_max_smoothing")

        self.contrastive_dim = config.get("contrastive_dim")

    def transform_embed_dim(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Transform the dimension of the input tensor from `embeds.size(1)` to `self.contrastive_dim`.

        Args:
            embeds (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.

        If the dimension of the input tensor is already equal to `self.contrastive_dim`, the input tensor is returned directly.
        Otherwise, a `nn.Linear` module is created to transform the dimension of the input tensor to `self.contrastive_dim`,
        and the input tensor is passed to the module for transformation. The transformed tensor is then returned.

        """
        if embeds.size(1) != self.contrastive_dim:
            embeds_dim_transformer = nn.Linear(
                embeds.size(1), self.contrastive_dim
            )
            embeds = embeds_dim_transformer(embeds)
        return embeds

    def forward(self, x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:

        concepts, concept_cosine_similarity = self.concepts(self.norm_concepts)

        image_embeds = torch.flatten(
            self.image_encoder(x), start_dim=1
        )
        assert self.image_dim == image_embeds.size(1)
        image_embeds, image_concept_attention_weight = self.conceptualizer(
            image_embeds, concepts
        )
        assert self.contrastive_dim == image_embeds.size(1)

        text_embeds = self.text_encoder(classes_idx)
        text_embeds = self.transform_embed_dim(embeds=text_embeds)

        outputs = self.contrast(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        return {
            "outputs": outputs,
            "image_low_concept_attention_weight": image_concept_attention_weight,
            "low_concept_cosine_similarity": concept_cosine_similarity
        }


MODELS = OrderedDict(
    {
        "OriTextResNet": OriTextResNet,
        "OriTextConceptualResNet": OriTextConceptualResNet
    }
)
