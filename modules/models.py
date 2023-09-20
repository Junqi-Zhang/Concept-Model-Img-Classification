import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torch import Tensor
from typing import Callable, Dict
from collections import OrderedDict
from sparsemax import Sparsemax


class BaseResNet(nn.Module):
    """
    Base class for ResNet models.

    Args:
        backbone_fn (Callable): Function to create the backbone model.
        num_classes (int): Number of output classes.
    """

    def __init__(self, backbone_fn: Callable, num_classes: int):
        super(BaseResNet, self).__init__()
        self.backbone = backbone_fn(weights=None, num_classes=num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing the model's output.
        """
        return {"outputs": self.backbone(x)}


class ResNet18(BaseResNet):
    """
    ResNet18 model.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes: int, *args, **kwargs):
        super(ResNet18, self).__init__(resnet18, num_classes)


class ResNet50(BaseResNet):
    """
    ResNet50 model.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes: int, *args, **kwargs):
        super(ResNet50, self).__init__(resnet50, num_classes)


class ContrastiveImgClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(ContrastiveImgClassifier, self).__init__()

        self.post_layernorm = nn.LayerNorm(input_dim)

        # Learnable mapping matrix from image space to joint image-text space
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

    def init_parameters(self) -> None:
        # Initialize the mapping matrix from image space to joint image-text space
        nn.init.xavier_uniform_(self.image_projection)
        # Initialize clf
        nn.init.xavier_uniform_(self.clf)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.post_layernorm(x)  # Align with CLIP post_layernorm
        image_embeds = torch.matmul(
            x, self.image_projection
        )  # B * D, align with CLIP mapping to image-text space

        # Normalize image_embeds using L2 norm, align with CLIP
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # Add noise to classification weights, align with CLIP
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # Normalize clf using L2 norm
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

        return {"outputs": outputs}


class ContrastiveResNet18(nn.Module):
    def __init__(self, num_classes: int, *args, **kwargs):
        super(ContrastiveResNet18, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.clf = ContrastiveImgClassifier(
            input_dim=512,
            num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        return self.clf(x)


class BasicConceptQuantizationV4(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float
    ):
        super(BasicConceptQuantizationV4, self).__init__()

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
        self.sparsemax = Sparsemax(dim=1)

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

        # Linear transformations for query, key, and value
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # Use sparsemax

        concept_summary = torch.matmul(
            attention_weights * self.grad_factor, value
        )  # B * D
        concept_summary = self.post_layernorm(concept_summary)  # B * D

        image_embeds = torch.matmul(
            concept_summary, self.image_projection
        )  # B * D
        # Normalize image_embeds using L2 norm
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # Add noise to classification weights
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # Normalize clf using L2 norm
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

        # Calculate the cosine similarity matrix for concepts
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


class BasicQuantResNet18V4(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float,
        *args,
        **kwargs
    ):
        super(BasicQuantResNet18V4, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV4(
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


class BasicQuantResNet50V4(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float,
        *args,
        **kwargs
    ):
        super(BasicQuantResNet50V4, self).__init__()

        img_classifier = resnet50(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        # Add a fully connected layer to map the dimension from 2048 to 512
        self.fc = nn.Linear(2048, 512)

        self.cq = BasicConceptQuantizationV4(
            input_dim=512,  # Keep it as 512, as we're mapping the output of ResNet50 to a 512-dimensional vector
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            grad_factor=grad_factor
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 2048-dimensional vector for ResNet50
        x = self.fc(x)  # Map the dimension from 2048 to 512
        return self.cq(x)


class BasicConceptQuantizationV4Smooth(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float,
        smoothing: float
    ):
        super(BasicConceptQuantizationV4Smooth, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts
        self.grad_factor = grad_factor
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
        self.sparsemax = Sparsemax(dim=1)

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
        attention_weights = self.sparsemax(attention_weights)  # Use sparsemax

        def smooth_tensor_matrix(input_matrix: torch.Tensor, smoothing=0.1):
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
            attention_weights_applied * self.grad_factor, value
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


class BasicQuantResNet18V4Smooth(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        norm_concepts: bool,
        grad_factor: float,
        smoothing: float,
        *args,
        **kwargs
    ):
        super(BasicQuantResNet18V4Smooth, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV4Smooth(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            grad_factor=grad_factor,
            smoothing=smoothing
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512-dimensional vector for ResNet18
        return self.cq(x)


MODELS = OrderedDict(
    {
        "ResNet18": ResNet18,
        "ResNet50": ResNet50,
        "ContrastiveResNet18": ContrastiveResNet18,
        "BasicQuantResNet18V4": BasicQuantResNet18V4,
        "BasicQuantResNet50V4": BasicQuantResNet50V4,
        "BasicQuantResNet18V4Smooth": BasicQuantResNet18V4Smooth
    }
)
