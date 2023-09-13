import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from collections import OrderedDict
from sparsemax import Sparsemax


class ResNet18(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return {"outputs": self.backbone(x)}


class ResNet50(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super(ResNet50, self).__init__()

        self.backbone = resnet50(weights=None, num_classes=num_classes)

    def forward(self, x):
        return {"outputs": self.backbone(x)}


class BasicConceptQuantizationV3(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, norm_summary, grad_factor):
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

        # W_q 和 W_k 设置为可学习参数
        self.query_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D
        self.key_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 设置 W_v 为单位阵
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # 参数初始化
        self.init_parameters()

        # attention_weight sparsemax
        self.sparsemax = Sparsemax(dim=1)

        # 分类层
        self.fc = nn.Linear(input_dim, num_classes)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)
        # 初始化 W_q 和 W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)

    def forward(self, x):

        # The shape of x should be B * D,
        # where B represents the batch_size.

        if self.norm_concepts:
            concepts = torch.div(
                self.concepts,
                torch.norm(self.concepts, dim=1, p=2).view(-1, 1)
            )
        else:
            concepts = self.concepts

        # query, key, value 的线性变换
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # 尝试sparsemax

        concept_summary = torch.matmul(
            attention_weights * self.grad_factor, value
        )  # B * D
        if self.norm_summary:
            # 按L2范数对concept_summary进行归一化
            concept_summary = torch.div(
                concept_summary,
                torch.norm(concept_summary, dim=1, p=2).view(-1, 1)
            )

        # The shape of output is B * K,
        # where K represents num_classes.
        outputs = self.fc(concept_summary)

        # 计算 concepts 的 cosine 相似度矩阵
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


class BasicConceptQuantizationV4(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, grad_factor):
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

        # W_q 和 W_k 设置为可学习参数
        self.query_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D
        self.key_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 设置 W_v 为单位阵
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # 图像空间到图文联合空间的可学习映射矩阵
        self.image_projection = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 分类层
        self.clf = nn.Parameter(
            torch.Tensor(num_classes, input_dim)
        )  # K * D, K represents the num_classes

        # scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # 参数初始化
        self.init_parameters()

        # attention_weight sparsemax
        self.sparsemax = Sparsemax(dim=1)

        # normalization layers
        # self.pre_layernorm = nn.LayerNorm(input_dim)
        # self.concept_layernorm = nn.LayerNorm(input_dim)
        self.post_layernorm = nn.LayerNorm(input_dim)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)
        # 初始化 W_q 和 W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)
        # 初始化图像空间到图文联合空间的映射矩阵
        nn.init.xavier_uniform_(self.image_projection)
        # 初始化 clf
        nn.init.xavier_uniform_(self.clf)

    def forward(self, x):

        # The shape of x should be B * D,
        # where B represents the batch_size.

        if self.norm_concepts:
            concepts = torch.div(
                self.concepts,
                torch.norm(self.concepts, dim=1, p=2).view(-1, 1)
            )
        else:
            concepts = self.concepts

        # query, key, value 的线性变换
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # 使用sparsemax

        concept_summary = torch.matmul(
            attention_weights * self.grad_factor, value
        )  # B * D
        concept_summary = self.post_layernorm(concept_summary)  # B * D

        image_embeds = torch.matmul(
            concept_summary, self.image_projection
        )  # B * D
        # 按L2范数对 image_embeds 进行归一化
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # 给分类权重增加噪声
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # 按L2范数对 clf 进行归一化
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

        # 计算 concepts 的 cosine 相似度矩阵
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


class BasicQuantResNet18V3(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, norm_summary, grad_factor, *args, **kwargs):
        super(BasicQuantResNet18V3, self).__init__()

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

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class ContrastiveImgClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ContrastiveImgClassifier, self).__init__()

        self.post_layernorm = nn.LayerNorm(input_dim)

        # 图像空间到图文联合空间的可学习映射矩阵
        self.image_projection = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 分类层
        self.clf = nn.Parameter(
            torch.Tensor(num_classes, input_dim)
        )  # K * D, K represents the num_classes

        # scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        # 初始化图像空间到图文联合空间的映射矩阵
        nn.init.xavier_uniform_(self.image_projection)
        # 初始化 clf
        nn.init.xavier_uniform_(self.clf)

    def forward(self, x):
        x = self.post_layernorm(x)  # 对齐 CLIP post_layernorm
        image_embeds = torch.matmul(
            x, self.image_projection
        )  # B * D, 对齐 CLIP 向图文空间映射

        # 对齐 CLIP 按L2范数对 image_embeds 进行归一化
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # 对齐 CLIP 给分类权重增加噪声
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # 按L2范数对 clf 进行归一化
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
    def __init__(self, num_classes, *args, **kwargs):
        super(ContrastiveResNet18, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.clf = ContrastiveImgClassifier(
            input_dim=512,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.clf(x)


class BasicQuantResNet18V4(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts=True, grad_factor=1, *args, **kwargs):
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

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class BasicConceptQuantizationV4Smooth(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, grad_factor, smoothing):
        super(BasicConceptQuantizationV4Smooth, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts
        self.grad_factor = grad_factor
        self.smoothing = smoothing  # sparsemax smoothing 的强度, (可考虑联动稀疏性正则权重)

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # W_q 和 W_k 设置为可学习参数
        self.query_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D
        self.key_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 设置 W_v 为单位阵
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # 图像空间到图文联合空间的可学习映射矩阵
        self.image_projection = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 分类层
        self.clf = nn.Parameter(
            torch.Tensor(num_classes, input_dim)
        )  # K * D, K represents the num_classes

        # scaler
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        # 参数初始化
        self.init_parameters()

        # attention_weight sparsemax
        self.sparsemax = Sparsemax(dim=1)

        # normalization layers
        # self.pre_layernorm = nn.LayerNorm(input_dim)
        # self.concept_layernorm = nn.LayerNorm(input_dim)
        self.post_layernorm = nn.LayerNorm(input_dim)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)
        # 初始化 W_q 和 W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)
        # 初始化图像空间到图文联合空间的映射矩阵
        nn.init.xavier_uniform_(self.image_projection)
        # 初始化 clf
        nn.init.xavier_uniform_(self.clf)

    def forward(self, x):

        # The shape of x should be B * D,
        # where B represents the batch_size.

        if self.norm_concepts:
            concepts = torch.div(
                self.concepts,
                torch.norm(self.concepts, dim=1, p=2).view(-1, 1)
            )
        else:
            concepts = self.concepts

        # query, key, value 的线性变换
        query = torch.matmul(x, self.query_transform)  # B * D
        key = torch.matmul(concepts, self.key_transform)  # C * D
        value = torch.matmul(concepts, self.value_transform)  # C * D

        attention_weights = torch.matmul(query, key.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.sparsemax(attention_weights)  # 使用sparsemax
        # attention_weights = F.softmax(attention_weights, dim=1)  # 使用softmax

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

        # train 和 eval 模式不同的前向传递结构
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
        # 按L2范数对 image_embeds 进行归一化
        image_embeds = torch.div(
            image_embeds,
            torch.norm(image_embeds, dim=1, p=2).view(-1, 1)
        )  # B * D

        # 给分类权重增加噪声
        clf = self.clf + (torch.rand_like(self.clf) - 0.5) * 0.01
        # 按L2范数对 clf 进行归一化
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

        # 计算 concepts 的 cosine 相似度矩阵
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


class BasicQuantResNet18V4Smooth(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, grad_factor, smoothing, *args, **kwargs):
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

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class BasicQuantResNet50V4(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts=True, grad_factor=1, *args, **kwargs):
        super(BasicQuantResNet50V4, self).__init__()

        img_classifier = resnet50(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.fc = nn.Linear(2048, 512)  # 添加全连接层，将维度从2048映射到512

        self.cq = BasicConceptQuantizationV4(
            input_dim=512,  # 保持为512，因为我们将ResNet50的输出映射到了512维
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            grad_factor=grad_factor
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 2048维向量 for ResNet50
        x = self.fc(x)  # 将维度从2048映射到512
        return self.cq(x)


MODELS = OrderedDict(
    {
        "ResNet18": ResNet18,
        "ContrastiveResNet18": ContrastiveResNet18,
        "BasicQuantResNet18V3": BasicQuantResNet18V3,
        "BasicQuantResNet18V4": BasicQuantResNet18V4,
        "BasicQuantResNet18V4Smooth": BasicQuantResNet18V4Smooth,
        "ResNet50": ResNet50,
        "BasicQuantResNet50V4": BasicQuantResNet50V4
    }
)
