import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torch import Tensor
from typing import Callable, Dict
from collections import OrderedDict
from sparsemax import Sparsemax


class BasicConceptQuantizationV3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        norm_summary: bool,
        grad_factor: float
    ):
        super(BasicConceptQuantizationV3, self).__init__()

        self.input_dim = input_dim
        self.grad_factor = grad_factor
        self.norm_concepts = norm_concepts
        self.norm_summary = norm_summary

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

        # Parameter initialization
        self.init_parameters()

        # attention_weight sparsemax
        self.sparsemax = Sparsemax(dim=1)

        # Classification layer
        self.fc = nn.Linear(input_dim, num_classes)

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

        # Linear transformation of query, key, and value
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # Try sparsemax

        concept_summary = torch.matmul(
            attention_weights * self.grad_factor, value
        )  # B * D
        if self.norm_summary:
            # Normalize concept_summary by L2 norm
            concept_summary = torch.div(
                concept_summary,
                torch.norm(concept_summary, dim=1, p=2).view(-1, 1)
            )

        # The shape of output is B * K,
        # where K represents num_classes.
        outputs = self.fc(concept_summary)

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
            "outputs": outputs,
            "attention_weights": attention_weights,
            "concept_similarity": concept_similarity
        }


class BasicQuantResNet18V3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        norm_summary: bool,
        grad_factor: float,
        *args,
        **kwargs
    ):
        super(BasicQuantResNet18V3, self).__init__()

        # Initialize the image classifier with ResNet18 architecture
        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV3(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            norm_summary=norm_summary,
            grad_factor=grad_factor
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        return self.cq(x)


MODELS = OrderedDict(
    {
        "BasicQuantResNet18V3": BasicQuantResNet18V3
    }
)
