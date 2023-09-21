import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch import Tensor
from typing import Dict
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

    def forward(self) -> Tensor:
        """
        Linearly transform the fixed text embeddings to the specified dimension.

        Returns:
            Tensor: The transformed text embeddings.
        """
        return self.linear(self.text_embeds)


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

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        image_dict = self.image_cq(x)
        image_embeds = torch.matmul(
            self.image_post_layernorm(image_dict["concept_summary"]),
            self.image_projection
        )
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

        text_embeds = torch.matmul(
            self.text_post_layernorm(self.text_encoder()),
            self.text_projection
        )
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # The shape of output is B * K,
        # where K represents num_classes.
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, text_embeds.t()
        ) * logit_scale  # B * K

        return {
            "outputs": outputs,
            "attention_weights": image_dict["attention_weights"],
            "concept_similarity": image_dict["concept_similarity"]
        }


MODELS_EXP = OrderedDict(
    {
        "SmoothConceptQuantization": SmoothConceptQuantization,
        "BasicTextEncoder": BasicTextEncoder,
        "OriTextQuantResNet18": OriTextQuantResNet18,
    }
)
