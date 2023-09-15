import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18
from typing import Dict
from collections import OrderedDict


class BasicConceptQuantizationV4NoSparse(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float
    ):
        super(BasicConceptQuantizationV4NoSparse, self).__init__()

        self.input_dim = input_dim
        self.grad_factor = grad_factor
        self.norm_concepts = norm_concepts

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # Set W_q and W_k as learnable parameters
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

        # Learnable mapping matrix from image space to image-text joint space
        self.image_projection = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # Classification layer
        self.clf = nn.Parameter(
            torch.Tensor(num_classes, input_dim)
        )  # K * D, K represents the num_classes

        # Scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # Parameter initialization
        self.init_parameters()

        # Attention_weight sparsemax
        # self.sparsemax = Sparsemax(dim=1)

        # Normalization layers
        # self.pre_layernorm = nn.LayerNorm(input_dim)
        # self.concept_layernorm = nn.LayerNorm(input_dim)
        self.post_layernorm = nn.LayerNorm(input_dim)

    def init_parameters(self) -> None:
        # Initialize concepts
        nn.init.xavier_uniform_(self.concepts)
        # Initialize W_q and W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)
        # Initialize the mapping matrix from image space to image-text joint space
        nn.init.xavier_uniform_(self.image_projection)
        # Initialize clf
        nn.init.xavier_uniform_(self.clf)

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
        # attention_weights = self.sparsemax(attention_weights)  # Use sparsemax
        attention_weights = F.softmax(attention_weights, dim=1)  # Use softmax

        concept_summary = torch.matmul(
            attention_weights * self.grad_factor, value
        )  # B * D
        concept_summary = self.post_layernorm(concept_summary)  # B * D

        image_embeds = torch.matmul(
            concept_summary, self.image_projection
        )  # B * D
        # Normalize image_embeds by L2 norm
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # Add noise to classification weights
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # Normalize clf by L2 norm
        clf = torch.div(
            clf,
            torch.norm(clf, dim=1, p=2).view(-1, 1)
        )  # K * D

        # The shape of output is B * K,
        # where K represents num_classes.
        logit_scale = self.logit_scale.exp()
        outputs = torch.matmul(
            image_embeds, clf.t()
        ) * logit_scale  # B * K

        # Calculate the cosine similarity matrix of concepts
        concept_similarity = F.cosine_similarity(
            concepts.unsqueeze(1),
            concepts.unsqueeze(0),
            dim=2
        )  # C * C

        return {
            "outputs": outputs,
            "attention_weights": attention_weights,
            "concept_similarity": concept_similarity
        }


class BasicQuantResNet18V4NoSparse(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float,
        *args,
        **kwargs,
    ):
        super(BasicQuantResNet18V4NoSparse, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV4NoSparse(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            grad_factor=grad_factor
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        return self.cq(x)


MODELS_EXP = OrderedDict(
    {
        "BasicQuantResNet18V4NoSparse": BasicQuantResNet18V4NoSparse
    }
)
