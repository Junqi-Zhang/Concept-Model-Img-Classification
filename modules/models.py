import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sparsemax import Sparsemax
from .utils import Recorder
from typing import Callable, Dict, List, Tuple, Any
from collections import OrderedDict


def normalize_rows(input_tensor: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Normalize the rows of a tensor.

    Args:
        input_tensor: The input tensor to be normalized.
        epsilon: A small value added to the row sums to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    input_tensor = input_tensor.to(torch.float)
    row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
    row_sums += epsilon
    normalized_tensor = input_tensor / row_sums
    return normalized_tensor


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

        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.xavier_uniform_(self.text_projection.weight)

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
    """
    A PyTorch module that combines an image encoder and a text encoder to perform contrastive learning.

    Args:
        config (Recorder): A configuration object that contains the following attributes:
            - backbone_name (str): The name of the image encoder backbone.
            - image_dim (int): The dimension of the image embeddings.
            - text_embeds_path (str): The path to the text embeddings file.
            - text_dim (int): The dimension of the text embeddings.
            - contrastive_dim (int): The dimension of the contrastive embeddings.

    Attributes:
        backbone_callable (Callable): A callable that returns the image encoder backbone.
        backbone (nn.Module): The image encoder backbone.
        image_encoder (nn.Sequential): The image encoder.
        text_encoder (TextEncoderSimulator): The text encoder.
        image_dim_transformer (nn.Linear): A linear layer that transforms the image embeddings to the contrastive dimension.
        text_dim_transformer (nn.Linear): A linear layer that transforms the text embeddings to the contrastive dimension.
        contrast (ContrastImageTextEmbeds): A module that performs contrastive learning on the image and text embeddings.

    Methods:
        parse_config(config: Recorder) -> None: Parses the configuration object.
        build_transform_dim_layer() -> None: Builds the linear layers that transform the image and text embeddings to the contrastive dimension.
        forward(x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]: Performs a forward pass of the model.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the model outputs.
    """

    def __init__(self, config: Recorder):
        super(OriTextResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=1
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-1]
        )
        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.build_transform_dim_layer()

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.image_dim = config.get("image_dim")

        self.text_embeds_path = config.get("text_embeds_path")
        self.text_dim = config.get("text_dim")

        self.contrastive_dim = config.get("contrastive_dim")

    def build_transform_dim_layer(self) -> None:
        if self.image_dim != self.contrastive_dim:
            self.image_dim_transformer = nn.Linear(
                self.image_dim, self.contrastive_dim
            )
        if self.text_dim != self.contrastive_dim:
            self.text_dim_transformer = nn.Linear(
                self.text_dim, self.contrastive_dim
            )

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
        if self.image_dim != self.contrastive_dim:
            image_embeds = self.image_dim_transformer(image_embeds)

        text_embeds = self.text_encoder(classes_idx)
        if self.text_dim != self.contrastive_dim:
            text_embeds = self.text_dim_transformer(text_embeds)

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
            self.q_linear = nn.Linear(query_dim, key_dim*n_head, bias=False)
            self.k_linear = nn.Linear(key_dim, key_dim*n_head, bias=False)
        else:
            assert key_dim % n_head == 0
            self.d_head = key_dim // n_head
            self.q_linear = nn.Linear(query_dim, key_dim, bias=False)
            self.k_linear = nn.Linear(key_dim, key_dim, bias=False)

        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)

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
    """
    A PyTorch module that combines a ResNet-based image encoder and a text encoder
    to produce contrastive embeddings for images and their corresponding textual
    descriptions.

    Args:
        config (Recorder): A configuration object that contains hyperparameters
            for the model.

    Attributes:
        backbone_callable (Callable): A callable that returns a ResNet-based
            backbone network.
        backbone (nn.Module): A ResNet-based backbone network.
        image_encoder (nn.Module): A ResNet-based image encoder.
        text_encoder (TextEncoderSimulator): A text encoder that produces
            embeddings for textual descriptions.
        concepts (Concepts): A module that generates low-level concepts from
            image features.
        conceptualizer (Conceptualizer): A module that maps image features to
            low-level concepts.
        contrast (ContrastImageTextEmbeds): A module that produces contrastive
            embeddings for images and text.

    Methods:
        parse_config(config: Recorder) -> None: Parses the configuration object
            and sets the model's attributes.
        build_transform_dim_layer() -> None: Builds a linear layer to transform
            text embeddings to the contrastive dimension.
        forward(x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:
            Computes the contrastive embeddings for a batch of images and their
            corresponding textual descriptions.

    """

    def __init__(self, config: Recorder):
        super(OriTextConceptualResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=1
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-1]
        )

        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.build_transform_dim_layer()

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
        self.text_dim = config.get("text_dim")

        self.concept_dim = config.get("concept_dim")
        self.num_concepts = config.get("num_low_concepts")
        self.norm_concepts = config.get("norm_low_concepts")

        self.n_head = config.get("image_low_concept_num_heads")
        self.keep_head_dim = config.get("image_low_concept_keep_head_dim")
        self.max_function_name = config.get("image_low_concept_max_function")
        self.max_smoothing = config.get("image_low_concept_max_smoothing")

        self.contrastive_dim = config.get("contrastive_dim")

    def build_transform_dim_layer(self) -> None:
        if self.text_dim != self.contrastive_dim:
            self.text_dim_transformer = nn.Linear(
                self.text_dim, self.contrastive_dim
            )

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
        if self.text_dim != self.contrastive_dim:
            text_embeds = self.text_dim_transformer(text_embeds)

        outputs = self.contrast(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        return {
            "outputs": outputs,
            "image_low_concept_attention_weight": image_concept_attention_weight,
            "low_concept_cosine_similarity": concept_cosine_similarity
        }


class ConceptualPool2d(nn.Module):
    """
    A module that performs conceptual pooling on a 2D image tensor.

    Args:
        spacial_dim (int): The spatial dimension of the input image.
        feature_dim (int): The feature dimension of the input image.
        concept_dim (int): The dimension of the conceptual space.
        image_patch_n_head (int): The number of heads in the image patch attention mechanism.
        image_patch_keep_head_dim (bool): Whether to keep the head dimension in the image patch attention mechanism.
        image_patch_max_function_name (str): The name of the function used to compute the maximum in the image patch attention mechanism.
        image_patch_max_smoothing (float): The smoothing factor used in the image patch attention mechanism.
        patch_concept_n_head (int): The number of heads in the patch conceptual attention mechanism.
        patch_concept_keep_head_dim (bool): Whether to keep the head dimension in the patch conceptual attention mechanism.
        patch_concept_max_function_name (str): The name of the function used to compute the maximum in the patch conceptual attention mechanism.
        patch_concept_max_smoothing (float): The smoothing factor used in the patch conceptual attention mechanism.

    Inputs:
        patches (torch.Tensor): The input image tensor of shape [B, D_q, H, W].
        concepts (torch.Tensor): The input conceptual tensor of shape [B, N, D_kv].

    Outputs:
        conceptual_image (torch.Tensor): The output conceptual image tensor of shape [B, D_kv].
        image_concept_attention_weight (torch.Tensor): The attention weight tensor of shape [B, N].
        image_patch_attention_weight (torch.Tensor): The attention weight tensor of shape [B, 1+H*W].
        patch_concept_attention_weight (torch.Tensor): The attention weight tensor of shape [B, 1+H*W, N].
    """

    def __init__(self,
                 spacial_dim: int,
                 feature_dim: int,
                 concept_dim: int,
                 image_patch_n_head: int,
                 image_patch_keep_head_dim: bool,
                 image_patch_max_function_name: str,
                 image_patch_max_smoothing: float,
                 patch_concept_n_head: int,
                 patch_concept_keep_head_dim: bool,
                 patch_concept_max_function_name: str,
                 patch_concept_max_smoothing: float):
        super(ConceptualPool2d, self).__init__()

        # positional embedding initialization
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, feature_dim) / feature_dim ** 0.5
        )
        # image spatial attention
        self.image_patch_attention = ModifiedMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=feature_dim,
            n_head=image_patch_n_head,
            keep_head_dim=image_patch_keep_head_dim,
            max_function_name=image_patch_max_function_name,
            max_smoothing=image_patch_max_smoothing
        )

        # patch conceptual attention
        self.patch_concept_attention = ModifiedMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=concept_dim,
            n_head=patch_concept_n_head,
            keep_head_dim=patch_concept_keep_head_dim,
            max_function_name=patch_concept_max_function_name,
            max_smoothing=patch_concept_max_smoothing
        )

    def forward(self, patches: torch.Tensor, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # reshape patches [B, D_q, H, W] to [B, H*W, D_q]
        patches = patches.flatten(start_dim=2).permute(0, 2, 1)
        patches = torch.cat(
            [patches.mean(dim=1, keepdim=True), patches],
            dim=1
        )  # [B, 1+H*W, D_q]

        # patch conceptual attention
        concepts = concepts.unsqueeze(0)  # [1, N, D_kv]
        conceptual_patches, patch_concept_attention_weight = self.patch_concept_attention(
            patches, concepts, concepts
        )  # [B, 1+H*W, D_kv], [B, 1+H*W, N]

        # image spatial attention
        # reshape positional_embedding [1+H*W, D_q] to [1, 1+H*W, D_q]
        positional_embedding = self.positional_embedding.unsqueeze(0)
        patches = patches + positional_embedding
        conceptual_image, image_patch_attention_weight = self.image_patch_attention(
            patches[:, :1], patches, conceptual_patches
        )  # [B, 1, D_kv], [B, 1, 1+H*W]

        conceptual_image = conceptual_image.squeeze(1)  # [B, D_kv]
        image_concept_attention_weight = torch.matmul(
            image_patch_attention_weight, patch_concept_attention_weight
        ).squeeze(1)  # [B, N]
        image_patch_attention_weight = image_patch_attention_weight.squeeze(
            1)  # [B, 1+H*W]

        return (
            conceptual_image,  # [B, D_kv]
            image_concept_attention_weight,  # [B, N]
            image_patch_attention_weight,  # [B, 1+H*W]
            patch_concept_attention_weight  # [B, 1+H*W, N]
        )


class OriTextConceptPoolResNet(nn.Module):
    """
    A PyTorch model class that implements a ResNet-based image encoder with a
    text encoder, a concept encoder, a concept pooling layer, and a contrastive
    loss layer.

    Args:
        config (Recorder): A configuration object that contains the
            hyperparameters and settings for the model.

    Attributes:
        backbone_callable (callable): A callable object that returns a
            ResNet-based backbone.
        backbone (nn.Module): A ResNet-based backbone that extracts image
            features.
        image_encoder (nn.Sequential): A sequential module that contains the
            layers of the ResNet-based backbone, except the last two layers.
        text_encoder (TextEncoderSimulator): A text encoder that encodes class
            labels into text embeddings.
        concepts (Concepts): A concept encoder that encodes low-level concepts
            into concept embeddings.
        conceptual_pooling (ConceptualPool2d): A concept pooling layer that
            pools image patches into image embeddings.
        contrast (ContrastImageTextEmbeds): A contrastive loss layer that
            computes the similarity between image and text embeddings.

    Methods:
        parse_config(config: Recorder) -> None:
            Parses the configuration object and initializes the hyperparameters
            and settings for the model.
        build_transform_dim_layer() -> None:
            Builds a linear layer that transforms the dimension of the text
            embeddings to match the input dimension of the contrastive loss
            layer.
        forward(x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:
            Computes the forward pass of the model given an input tensor and a
            list of class indices, and returns a dictionary of outputs,
            including the model's predictions, concept attention weights,
            patch attention weights, and concept similarity scores.
    """

    def __init__(self, config: Recorder):
        super(OriTextConceptPoolResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=1
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.build_transform_dim_layer()

        self.concepts = Concepts(
            num_concepts=self.num_concepts,
            concept_dim=self.concept_dim
        )

        self.conceptual_pooling = ConceptualPool2d(
            spacial_dim=self.spacial_dim,
            feature_dim=self.image_dim,
            concept_dim=self.concept_dim,
            image_patch_n_head=self.image_patch_n_head,
            image_patch_keep_head_dim=self.image_patch_keep_head_dim,
            image_patch_max_function_name=self.image_patch_max_function_name,
            image_patch_max_smoothing=self.image_patch_max_smoothing,
            patch_concept_n_head=self.patch_concept_n_head,
            patch_concept_keep_head_dim=self.patch_concept_keep_head_dim,
            patch_concept_max_function_name=self.patch_concept_max_function_name,
            patch_concept_max_smoothing=self.patch_concept_max_smoothing
        )

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.image_dim = config.get("image_dim")
        self.spacial_dim = config.get("image_spacial_dim")

        self.text_embeds_path = config.get("text_embeds_path")
        self.text_dim = config.get("text_dim")

        self.concept_dim = config.get("concept_dim")
        self.num_concepts = config.get("num_low_concepts")
        self.norm_concepts = config.get("norm_low_concepts")

        self.image_patch_n_head = config.get("image_patch_num_heads")
        self.image_patch_keep_head_dim = config.get(
            "image_patch_keep_head_dim")
        self.image_patch_max_function_name = config.get(
            "image_patch_max_function")
        self.image_patch_max_smoothing = config.get(
            "image_patch_max_smoothing")

        self.patch_concept_n_head = config.get("patch_low_concept_num_heads")
        self.patch_concept_keep_head_dim = config.get(
            "patch_low_concept_keep_head_dim")
        self.patch_concept_max_function_name = config.get(
            "patch_low_concept_max_function")
        self.patch_concept_max_smoothing = config.get(
            "patch_low_concept_max_smoothing")

        self.contrastive_dim = config.get("contrastive_dim")

    def build_transform_dim_layer(self) -> None:
        if self.text_dim != self.contrastive_dim:
            self.text_dim_transformer = nn.Linear(
                self.text_dim, self.contrastive_dim
            )

    def forward(self, x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:
        concepts, concept_cosine_similarity = self.concepts(self.norm_concepts)

        image_patches = self.image_encoder(x)
        assert self.image_dim == image_patches.size(1)
        image_embeds, image_concept_attention_weight, image_patch_attention_weight, patch_concept_attention_weight = self.conceptual_pooling(
            image_patches, concepts
        )
        assert self.contrastive_dim == image_embeds.size(1)

        text_embeds = self.text_encoder(classes_idx)
        if self.text_dim != self.contrastive_dim:
            text_embeds = self.text_dim_transformer(text_embeds)

        outputs = self.contrast(
            image_embeds=image_embeds, text_embeds=text_embeds
        )

        return {
            "outputs": outputs,
            "image_low_concept_attention_weight": image_concept_attention_weight,
            "image_patch_attention_weight": image_patch_attention_weight,
            "patch_low_concept_attention_weight": patch_concept_attention_weight,
            "low_concept_cosine_similarity": concept_cosine_similarity
        }


class HierarchicalConcepts(nn.Module):
    """
    A PyTorch module for learning hierarchical concepts.

    Args:
        num_low_concepts (int): The number of low-level concepts.
        num_high_concepts (int): The number of high-level concepts.
        concept_dim (int): The dimensionality of the concept vectors.
        low_high_max_function (str): The max function of the low-to-high mapping.
        output_high_concepts_type (str): The type of output high-level concepts.
            Must be one of "original_high", "high_plus_low", or "aggregated_low".
        detach_low_concepts (bool): Whether to detach the low-level concepts.

    Attributes:
        low_concepts (Concepts): The low-level concepts module.
        high_concepts (Concepts): The high-level concepts module.
        concept_hierarchy_builder (ModifiedMultiHeadAttention): The module for
            building the hierarchy between low-level and high-level concepts.
        output_high_concepts_type (str): The type of output high-level concepts.

    Methods:
        forward(norm_low_concepts: bool, norm_high_concepts: bool) -> Tuple:
            Computes the forward pass of the hierarchical concepts module.

    """

    def __init__(self,
                 num_low_concepts: int,
                 num_high_concepts: int,
                 concept_dim: int,
                 low_high_max_function: str,
                 output_high_concepts_type: str,
                 detach_low_concepts: bool):
        super(HierarchicalConcepts, self).__init__()

        self.low_concepts = Concepts(
            num_concepts=num_low_concepts,
            concept_dim=concept_dim
        )
        self.high_concepts = Concepts(
            num_concepts=num_high_concepts,
            concept_dim=concept_dim
        )

        self.concept_hierarchy_builder = ModifiedMultiHeadAttention(
            query_dim=concept_dim,
            key_dim=concept_dim,
            n_head=1,
            keep_head_dim=True,
            max_function_name=low_high_max_function,
            max_smoothing=0.0
        )
        self.output_high_concepts_type = output_high_concepts_type
        self.detach_low_concepts = detach_low_concepts

    def forward(self, norm_low_concepts: bool, norm_high_concepts: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the hierarchical concepts module.

        Args:
            norm_low_concepts (bool): Whether to normalize the low-level concepts.
            norm_high_concepts (bool): Whether to normalize the high-level concepts.

        Returns:
            A tuple containing the following elements:
            - low_concepts (Tensor): The low-level concepts.
            - low_concept_cosine_similarity (Tensor): The cosine similarity between
              the low-level concepts and the input features.
            - output_high_concepts (Tensor): The output high-level concepts.
            - high_concept_cosine_similarity (Tensor): The cosine similarity between
              the high-level concepts and the input features.
            - low_high_hierarchy (Tensor): The hierarchy between low-level and
              high-level concepts.

        """
        low_concepts, low_concept_cosine_similarity = self.low_concepts(
            norm_low_concepts
        )
        high_concepts, high_concept_cosine_similarity = self.high_concepts(
            norm_high_concepts
        )

        if self.detach_low_concepts:
            high_related_low_concepts = low_concepts.detach()
        else:
            high_related_low_concepts = low_concepts

        _, low_high_hierarchy = self.concept_hierarchy_builder(
            high_related_low_concepts.unsqueeze(0),  # [1, N_low, D]
            high_concepts.unsqueeze(0),  # [1, N_high, D]
            high_concepts.unsqueeze(0)  # [1, N_high, D]
        )  # [1, N_low, N_high]

        low_high_hierarchy = low_high_hierarchy.squeeze(0)  # [N_low, N_high]

        if self.output_high_concepts_type == "original_high":
            output_high_concepts = high_concepts
        else:
            high_low_hierarchy = low_high_hierarchy.t()  # [N_high, N_low]
            output_high_concepts = torch.matmul(
                normalize_rows(high_low_hierarchy), high_related_low_concepts
            )  # [N_high, D]
            if self.output_high_concepts_type == "high_plus_low":
                output_high_concepts = output_high_concepts + high_concepts
            else:
                assert self.output_high_concepts_type == "aggregated_low"

        return (
            low_concepts,  # [N_low, D]
            low_concept_cosine_similarity,
            output_high_concepts,  # [N_high, D]
            high_concept_cosine_similarity,
            low_high_hierarchy  # [N_low, N_high]
        )


class HierarchicalConceptualPool2d(nn.Module):
    def __init__(self,
                 spacial_dim: int,
                 feature_dim: int,
                 concept_dim: int,
                 image_patch_n_head: int,
                 image_patch_keep_head_dim: bool,
                 image_patch_max_function_name: str,
                 image_patch_max_smoothing: float,
                 patch_concept_n_head: int,
                 patch_concept_keep_head_dim: bool,
                 patch_concept_max_function_name: str,
                 patch_concept_max_smoothing: float):
        super(HierarchicalConceptualPool2d, self).__init__()

        self.conceptual_pooling = ConceptualPool2d(
            spacial_dim=spacial_dim,
            feature_dim=feature_dim,
            concept_dim=concept_dim,
            image_patch_n_head=image_patch_n_head,
            image_patch_keep_head_dim=image_patch_keep_head_dim,
            image_patch_max_function_name=image_patch_max_function_name,
            image_patch_max_smoothing=image_patch_max_smoothing,
            patch_concept_n_head=patch_concept_n_head,
            patch_concept_keep_head_dim=patch_concept_keep_head_dim,
            patch_concept_max_function_name=patch_concept_max_function_name,
            patch_concept_max_smoothing=patch_concept_max_smoothing
        )

    def forward(self, patches: torch.Tensor, low_concepts: torch.Tensor, high_concepts: torch.Tensor, low_high_hierarchy: torch.Tensor):
        (
            low_conceptual_image,  # [B, D_kv]
            image_low_concept_attention_weight,  # [B, N_low]
            image_patch_attention_weight,  # [B, 1+H*W]
            patch_low_concept_attention_weight  # [B, 1+H*W, N_low]
        ) = self.conceptual_pooling(patches, low_concepts)

        image_high_concept_attention_weight = torch.matmul(
            image_low_concept_attention_weight, low_high_hierarchy
        )  # [B, N_high]
        patch_high_concept_attention_weight = torch.matmul(
            patch_low_concept_attention_weight, low_high_hierarchy
        )  # [B, 1+H*W, N_high]
        high_conceptual_image = torch.matmul(
            image_high_concept_attention_weight, high_concepts
        )  # [B, D_kv]

        return (
            low_conceptual_image,
            high_conceptual_image,
            image_patch_attention_weight,
            image_low_concept_attention_weight,
            image_high_concept_attention_weight,
            patch_low_concept_attention_weight,
            patch_high_concept_attention_weight
        )


class OriTextHierarchicalConceptualPoolResNet(nn.Module):
    def __init__(self, config: Recorder):
        super(OriTextHierarchicalConceptualPoolResNet, self).__init__()

        self.parse_config(config=config)

        self.backbone_callable = get_callable_backbone(self.backbone_name)
        self.backbone = self.backbone_callable(
            weights=None, num_classes=1
        )
        self.image_encoder = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        self.text_encoder = TextEncoderSimulator(
            text_embeds_path=self.text_embeds_path
        )

        self.build_transform_dim_layer()

        self.hierarchical_concepts = HierarchicalConcepts(
            num_low_concepts=self.num_low_concepts,
            num_high_concepts=self.num_high_concepts,
            concept_dim=self.concept_dim,
            low_high_max_function=self.low_high_max_function,
            output_high_concepts_type=self.output_high_concepts_type,
            detach_low_concepts=self.detach_low_concepts
        )

        self.hierarchical_conceptual_pooling = HierarchicalConceptualPool2d(
            spacial_dim=self.spacial_dim,
            feature_dim=self.image_dim,
            concept_dim=self.concept_dim,
            image_patch_n_head=self.image_patch_n_head,
            image_patch_keep_head_dim=self.image_patch_keep_head_dim,
            image_patch_max_function_name=self.image_patch_max_function_name,
            image_patch_max_smoothing=self.image_patch_max_smoothing,
            patch_concept_n_head=self.patch_concept_n_head,
            patch_concept_keep_head_dim=self.patch_concept_keep_head_dim,
            patch_concept_max_function_name=self.patch_concept_max_function_name,
            patch_concept_max_smoothing=self.patch_concept_max_smoothing
        )

        self.contrast = ContrastImageTextEmbeds(embed_dim=self.contrastive_dim)
        self.aux_contrast = ContrastImageTextEmbeds(
            embed_dim=self.contrastive_dim
        )

    def parse_config(self, config: Recorder) -> None:
        self.backbone_name = config.get("backbone_name")
        self.image_dim = config.get("image_dim")
        self.spacial_dim = config.get("image_spacial_dim")

        self.text_embeds_path = config.get("text_embeds_path")
        self.text_dim = config.get("text_dim")
        self.detach_text_embeds = config.get("detach_text_embeds")

        self.concept_dim = config.get("concept_dim")
        self.num_low_concepts = config.get("num_low_concepts")
        self.norm_low_concepts = config.get("norm_low_concepts")
        self.num_high_concepts = config.get("num_high_concepts")
        self.norm_high_concepts = config.get("norm_high_concepts")
        self.low_high_max_function = config.get("low_high_max_function")
        self.output_high_concepts_type = config.get(
            "output_high_concepts_type")
        self.detach_low_concepts = config.get("detach_low_concepts")

        self.image_patch_n_head = config.get("image_patch_num_heads")
        self.image_patch_keep_head_dim = config.get(
            "image_patch_keep_head_dim")
        self.image_patch_max_function_name = config.get(
            "image_patch_max_function")
        self.image_patch_max_smoothing = config.get(
            "image_patch_max_smoothing")

        self.patch_concept_n_head = config.get("patch_low_concept_num_heads")
        self.patch_concept_keep_head_dim = config.get(
            "patch_low_concept_keep_head_dim")
        self.patch_concept_max_function_name = config.get(
            "patch_low_concept_max_function")
        self.patch_concept_max_smoothing = config.get(
            "patch_low_concept_max_smoothing")

        self.contrastive_dim = config.get("contrastive_dim")

    def build_transform_dim_layer(self) -> None:
        if self.text_dim != self.contrastive_dim:
            self.text_dim_transformer = nn.Linear(
                self.text_dim, self.contrastive_dim
            )

    def forward(self, x: torch.Tensor, classes_idx: List) -> Dict[str, torch.Tensor]:
        (
            low_concepts,  # [N_low, D]
            low_concept_cosine_similarity,
            high_concepts,  # [N_high, D]
            high_concept_cosine_similarity,
            low_high_hierarchy  # [N_low, N_high]
        ) = self.hierarchical_concepts(
            norm_low_concepts=self.norm_low_concepts,
            norm_high_concepts=self.norm_high_concepts
        )

        image_patches = self.image_encoder(x)
        assert self.image_dim == image_patches.size(1)
        (
            low_conceptual_image,  # [B, D_kv]
            high_conceptual_image,  # [B, D_kv]
            image_patch_attention_weight,  # [B, 1+H*W]
            image_low_concept_attention_weight,  # [B, N_low]
            image_high_concept_attention_weight,  # [B, N_high]
            patch_low_concept_attention_weight,  # [B, 1+H*W, N_low]
            patch_high_concept_attention_weight  # [B, 1+H*W, N_high]
        ) = self.hierarchical_conceptual_pooling(
            image_patches, low_concepts, high_concepts, low_high_hierarchy
        )
        assert self.concept_dim == low_conceptual_image.size(1)
        assert self.concept_dim == high_conceptual_image.size(1)

        text_embeds = self.text_encoder(classes_idx)
        if self.text_dim != self.contrastive_dim:
            text_embeds = self.text_dim_transformer(text_embeds)
        outputs = self.contrast(
            image_embeds=low_conceptual_image,
            text_embeds=text_embeds
        )

        aux_text_embeds = text_embeds.detach() if self.detach_text_embeds else text_embeds
        aux_outputs = self.aux_contrast(
            image_embeds=high_conceptual_image,
            text_embeds=aux_text_embeds
        )

        return {
            "outputs": outputs,
            "aux_outputs": aux_outputs,
            "image_patch_attention_weight": image_patch_attention_weight,
            "image_low_concept_attention_weight": image_low_concept_attention_weight,
            "image_high_concept_attention_weight": image_high_concept_attention_weight,
            "patch_low_concept_attention_weight": patch_low_concept_attention_weight,
            "patch_high_concept_attention_weight": patch_high_concept_attention_weight,
            "low_concept_cosine_similarity": low_concept_cosine_similarity,
            "high_concept_cosine_similarity": high_concept_cosine_similarity,
            "low_high_hierarchy": low_high_hierarchy
        }


MODELS = OrderedDict(
    {
        "OriTextResNet": OriTextResNet,
        "OriTextConceptualResNet": OriTextConceptualResNet,
        "OriTextConceptPoolResNet": OriTextConceptPoolResNet,
        "OriTextHierarchicalConceptualPoolResNet": OriTextHierarchicalConceptualPoolResNet
    }
)
