import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from collections import OrderedDict
from sparsemax import Sparsemax


class ResNet18(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return {"outputs": self.backbone(x)}


class BasicConceptQuantization(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicConceptQuantization, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts
        self.norm_summary = norm_summary

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # W_q 和 W_k 暂时使用单位阵, 保留后续代码兼容性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D
        self.key_transform = torch.eye(
            input_dim,
            dtype=torch.float,
            device=device
        )  # D * D

        # 设置 W_v 为可学习参数
        self.value_transform = nn.Parameter(
            torch.Tensor(input_dim, input_dim)
        )  # D * D

        # 参数初始化
        self.init_parameters()

        # 分类层
        self.fc = nn.Linear(input_dim, num_classes)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)
        # 初始化 W_v
        nn.init.xavier_uniform_(self.value_transform)

    def modified_softmax(self, x, dim, modified=False):
        # 将输入转换为浮点型张量
        x = x.float()
        # 对每个元素减去输入张量的最大值，以提高数值稳定性
        x = x - torch.max(x, dim=dim, keepdim=True)[0]
        # 计算指数
        exp_x = torch.exp(x)
        # 沿着指定的维度进行求和
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        # 计算softmax
        if modified:
            softmax_x = exp_x / (sum_exp_x + 1)
        else:
            softmax_x = exp_x / sum_exp_x
        return softmax_x

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
        attention_weights = self.modified_softmax(
            attention_weights, dim=1, modified=False
        )  # 暂时使用原版softmax

        concept_summary = torch.matmul(attention_weights, value)  # B * D
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


class BasicConceptQuantizationV1(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicConceptQuantizationV1, self).__init__()

        self.input_dim = input_dim
        self.norm_concepts = norm_concepts
        self.norm_summary = norm_summary

        # The shape of self.concepts should be C * D,
        # where C represents the num_concepts,
        # D represents the input_dim.
        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, input_dim)
        )  # C * D

        # 参数初始化
        self.init_parameters()

        # 分类层
        self.fc = nn.Linear(input_dim, num_classes)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)

    def modified_softmax(self, x, dim, modified=False):
        # 将输入转换为浮点型张量
        x = x.float()
        # 对每个元素减去输入张量的最大值，以提高数值稳定性
        x = x - torch.max(x, dim=dim, keepdim=True)[0]
        # 计算指数
        exp_x = torch.exp(x)
        # 沿着指定的维度进行求和
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        # 计算softmax
        if modified:
            softmax_x = exp_x / (sum_exp_x + 1)
        else:
            softmax_x = exp_x / sum_exp_x
        return softmax_x

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

        attention_weights = torch.matmul(x, concepts.t())  # B * C
        attention_weights = attention_weights / \
            torch.sqrt(torch.tensor(self.input_dim).float())
        attention_weights = self.modified_softmax(
            attention_weights, dim=1, modified=False
        )  # 暂时使用原版softmax

        concept_summary = torch.matmul(
            attention_weights, concepts
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


class BasicConceptQuantizationV2(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicConceptQuantizationV2, self).__init__()

        self.input_dim = input_dim
        self.num_concepts = num_concepts
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

        # 分类层
        self.fc = nn.Linear(input_dim, num_classes)

    def init_parameters(self):
        # 初始化 concepts
        nn.init.xavier_uniform_(self.concepts)
        # 初始化 W_q 和 W_k
        nn.init.xavier_uniform_(self.query_transform)
        nn.init.xavier_uniform_(self.key_transform)

    def modified_softmax(self, x, dim, modified=False):
        # 将输入转换为浮点型张量
        x = x.float()
        # 对每个元素减去输入张量的最大值，以提高数值稳定性
        x = x - torch.max(x, dim=dim, keepdim=True)[0]
        # 计算指数
        exp_x = torch.exp(x)
        # 沿着指定的维度进行求和
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        # 计算softmax
        if modified:
            softmax_x = exp_x / (sum_exp_x + 1)
        else:
            softmax_x = exp_x / sum_exp_x
        return softmax_x

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
        attention_weights = self.modified_softmax(
            attention_weights, dim=1, modified=False
        )  # 暂时使用原版softmax

        concept_summary = torch.matmul(
            attention_weights * self.num_concepts, value
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


class BasicConceptQuantizationV3(nn.Module):
    def __init__(self, input_dim, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicConceptQuantizationV3, self).__init__()

        self.input_dim = input_dim
        self.num_concepts = num_concepts
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
            attention_weights * self.num_concepts, value
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


class BasicQuantResNet18(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicQuantResNet18, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantization(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            norm_summary=norm_summary
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class BasicQuantResNet18V1(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicQuantResNet18V1, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV1(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            norm_summary=norm_summary
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class BasicQuantResNet18V2(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicQuantResNet18V2, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV2(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            norm_summary=norm_summary
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


class BasicQuantResNet18V3(nn.Module):
    def __init__(self, num_classes, num_concepts, norm_concepts, norm_summary):
        super(BasicQuantResNet18V3, self).__init__()

        img_classifier = resnet18(weights=None, num_classes=num_classes)
        self.backbone = nn.Sequential(*list(img_classifier.children())[:-1])

        self.cq = BasicConceptQuantizationV3(
            input_dim=512,
            num_classes=num_classes,
            num_concepts=num_concepts,
            norm_concepts=norm_concepts,
            norm_summary=norm_summary
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 512维向量 for ResNet18
        return self.cq(x)


PROVIDED_MODELS = OrderedDict(
    {
        "ResNet18": ResNet18,
        "BasicQuantResNet18": BasicQuantResNet18,
        "BasicQuantResNet18V1": BasicQuantResNet18V1,
        "BasicQuantResNet18V2": BasicQuantResNet18V2,
        "BasicQuantResNet18V3": BasicQuantResNet18V3
    }
)
