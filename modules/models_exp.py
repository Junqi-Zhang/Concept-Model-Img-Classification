import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torch import Tensor
from typing import Dict, List
from collections import OrderedDict
from sparsemax import Sparsemax


class SmoothConceptQuantization(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        norm_concepts: bool,
        smoothing: float
    ):
        super(SmoothConceptQuantization, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts
        # Strength of sparsemax smoothing (consider linking sparsity regularization weight)
        self.smoothing = smoothing

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # W_q and W_k are set as learnable parameters
        self.query_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D
        self.key_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # Set W_v as the identity matrix
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # Parameter initialization
        self.init_parameters()

        # Attention_weight sparsemax
        self.sparsemax = Sparsemax(dim=1)

    def init_parameters(self) -> None:
        # Initialize concepts
        nn.init.xavier_uniform_(self.concepts)
        # Initialize W_q and W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:

        # The shape of x should be B * D,
        # where B represents the batch_size.

        if self.norm_concepts:
            concepts = torch.div(
                self.concepts,
                torch.norm(self.concepts, dim=1, p=2).view(-1, 1)
            )
        else:
            concepts = self.concepts

        # Linear transformation of query, key, value
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # Use sparsemax
        # attention_weights = F.softmax(attention_weights, dim=1)  # Use softmax

        def smooth_tensor_matrix(input_matrix: Tensor, smoothing=0.1):
            """
            Smooth every row in a tensor matrix in PyTorch
            """
            assert 0 <= smoothing < 1
            num_classes = input_matrix.size(1)
            non_zero_mask = input_matrix > 0
            num_nonzero_per_row = non_zero_mask.sum(dim=1, keepdim=True)
            num_zero_per_row = num_classes - num_nonzero_per_row
            smoothing_value_for_zeros = (
                smoothing / num_zero_per_row
            ).expand_as(input_matrix)
            smoothing_value_for_non_zeros = (
                smoothing / num_nonzero_per_row
            ).expand_as(input_matrix)
            smoothed_matrix = input_matrix.clone()
            smoothed_matrix[non_zero_mask] -= smoothing_value_for_non_zeros[non_zero_mask]
            smoothed_matrix[~non_zero_mask] += smoothing_value_for_zeros[~non_zero_mask]
            return smoothed_matrix

        # Different forward propagation structures for train and eval modes
        if self.training:
            attention_weights_applied = smooth_tensor_matrix(
                attention_weights, self.smoothing
            )
        else:
            attention_weights_applied = attention_weights

        concept_summary = torch.matmul(
            attention_weights_applied, value
        )  # B * D

        # Calculate the cosine similarity matrix of concepts
        if self.norm_concepts:
            normalized_concepts = concepts
        else:
            normalized_concepts = torch.div(
                concepts,
                torch.norm(concepts, dim=1, p=2).view(-1, 1)
            )
        concept_similarity = torch.matmul(
            normalized_concepts, normalized_concepts.t()
        )  # C * C

        return {
            "concept_summary": concept_summary,
            "attention_weights": attention_weights,
            "concept_similarity": concept_similarity
        }


class SoftConceptQuantization(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        norm_concepts: bool
    ):
        super(SoftConceptQuantization, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # W_q and W_k are set as learnable parameters
        self.query_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D
        self.key_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # Set W_v as the identity matrix
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize concepts
        nn.init.xavier_uniform_(self.concepts)
        # Initialize W_q and W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:

        # The shape of x should be B * D,
        # where B represents the batch_size.

        if self.norm_concepts:
            concepts = torch.div(
                self.concepts,
                torch.norm(self.concepts, dim=1, p=2).view(-1, 1)
            )
        else:
            concepts = self.concepts

        # Linear transformation of query, key, value
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = F.softmax(attention_weights, dim=1)  # Use softmax
        concept_summary = torch.matmul(
            attention_weights, value
        )  # B * D

        # Calculate the cosine similarity matrix of concepts
        if self.norm_concepts:
            normalized_concepts = concepts
        else:
            normalized_concepts = torch.div(
                concepts,
                torch.norm(concepts, dim=1, p=2).view(-1, 1)
            )
        concept_similarity = torch.matmul(
            normalized_concepts, normalized_concepts.t()
        )  # C * C

        return {
            "concept_summary": concept_summary,
            "attention_weights": attention_weights,
            "concept_similarity": concept_similarity
        }


class BasicTextEncoder(nn.Module):
    """
    BasicTextEncoder is a simple text encoder that linearly transforms fixed text embeddings to a specified dimension.
    """

    def __init__(self, text_embeds: Tensor, output_dim: int = 512):
        """
        Initialize BasicTextEncoder.

        Args:
            text_embeds (Tensor): Text embeddings obtained from another language model.
            output_dim (int, optional): The target dimension for the output. Defaults to 512.
        """
        super(BasicTextEncoder, self).__init__()
        self.text_embeds = nn.Parameter(text_embeds, requires_grad=False)
        self.linear = nn.Linear(text_embeds.size(1), output_dim)

    def forward(self, classes_idx: List) -> Tensor:
        """
        Linearly transform the fixed text embeddings to the specified dimension.

        Returns:
            Tensor: The transformed text embeddings.
        """
        return self.linear(self.text_embeds[classes_idx, :])


class OriTextQuantResNet18(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        norm_concepts: bool,
        concept_dim: int,
        smoothing: float,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextQuantResNet18, self).__init__()

        img_classifier = resnet18(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.image_cq = SmoothConceptQuantization(
            input_dim=concept_dim,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            smoothing=smoothing
        )
        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        image_dict = self.image_cq(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(image_dict["concept_summary"]),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs,
            "attention_weights": image_dict["attention_weights"],
            "concept_similarity": image_dict["concept_similarity"]
        }


class OriTextSoftQuantResNet18(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        norm_concepts: bool,
        concept_dim: int,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextSoftQuantResNet18, self).__init__()

        img_classifier = resnet18(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.image_cq = SoftConceptQuantization(
            input_dim=concept_dim,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts
        )
        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        image_dict = self.image_cq(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(image_dict["concept_summary"]),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs,
            "attention_weights": image_dict["attention_weights"],
            "concept_similarity": image_dict["concept_similarity"]
        }


class OriTextResNet18(nn.Module):
    def __init__(
        self,
        concept_dim: int,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextResNet18, self).__init__()

        img_classifier = resnet18(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        image_embeds = torch.matmul(
            self.image_post_layernorm(x),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs
        }


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(
            spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class OriTextModifiedResNet18(nn.Module):
    def __init__(
        self,
        concept_dim: int,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextModifiedResNet18, self).__init__()

        self.backbone = resnet18(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone.avgpool = AttentionPool2d(
            spacial_dim=7,
            embed_dim=concept_dim,
            num_heads=32
        )
        self.backbone.fc = nn.Identity()

        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(x),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs
        }


class OriTextQuantModifiedResNet18(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        norm_concepts: bool,
        concept_dim: int,
        smoothing: float,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextQuantModifiedResNet18, self).__init__()

        self.backbone = resnet18(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone.avgpool = AttentionPool2d(
            spacial_dim=7,
            embed_dim=concept_dim,
            num_heads=32
        )
        self.backbone.fc = nn.Identity()

        self.image_cq = SmoothConceptQuantization(
            input_dim=concept_dim,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            smoothing=smoothing
        )
        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        image_dict = self.image_cq(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(image_dict["concept_summary"]),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs,
            "attention_weights": image_dict["attention_weights"],
            "concept_similarity": image_dict["concept_similarity"]
        }


class OriTextResNet50(nn.Module):
    def __init__(
        self,
        concept_dim: int,
        text_embeds: Tensor,
        *args,
        **kwargs
    ):
        super(OriTextResNet50, self).__init__()

        img_classifier = resnet50(
            weights=None, num_classes=text_embeds.size(0)
        )
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])
        self.fc = nn.Linear(2048, concept_dim)

        self.image_post_layernorm = nn.LayerNorm(concept_dim)
        self.image_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        self.text_encoder = BasicTextEncoder(
            text_embeds=text_embeds,
            output_dim=concept_dim
        )
        self.text_post_layernorm = nn.LayerNorm(concept_dim)
        self.text_projection = nn.Parameter(
            torch.Tensor(concept_dim, concept_dim)
        )  # D * D

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image/text space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        nn.init.xavier_uniform_(self.text_projection)

    def forward(self, x: Tensor, classes_idx: List) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(x),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder(classes_idx)),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K = len(classes_idx).
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs
        }


MODELS_EXP = OrderedDict(
    {
        "OriTextQuantResNet18": OriTextQuantResNet18,
        "OriTextSoftQuantResNet18": OriTextSoftQuantResNet18,
        "OriTextResNet18": OriTextResNet18,
        "OriTextModifiedResNet18": OriTextModifiedResNet18,
        "OriTextQuantModifiedResNet18": OriTextQuantModifiedResNet18,
        "OriTextResNet50": OriTextResNet50,
    }
)
