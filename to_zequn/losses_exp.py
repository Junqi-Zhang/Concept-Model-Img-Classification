import torch
from torch import nn

# criterion_sparsity = SparsityLoss(
#     target=config.sparsity_target, gamma=config.gamma
# )
class SparsityLoss(nn.Module):
    def __init__(self, target, gamma, p=0.5, threshold=1e-7, epsilon=1e-7, dim=-1):
        super(SparsityLoss, self).__init__()

        self.target = target
        self.gamma = gamma
        self.p = p
        self.threshold = threshold
        self.epsilon = epsilon
        self.dim = dim

        self.cutoff = gamma ** p

    def forward(self, x):
        with torch.no_grad():
            switch = (torch.sum(x > self.threshold, dim=self.dim)
                      > self.target).type(x.dtype)

        loss_p = torch.pow(torch.abs(x) + self.epsilon, self.p)
        loss = torch.min(loss_p, self.cutoff *
                         torch.ones_like(x)).sum(self.dim)
        return (switch * loss).mean()

# criterion_orth = OrthogonalityLoss(config.num_concepts)
class OrthogonalityLoss(nn.Module):
    def __init__(self, n_component):
        super(OrthogonalityLoss, self).__init__()
        self.n_component = n_component
        self.ideal_similarity = torch.eye(
            n_component,
            dtype=torch.float
        )

    def forward(self, x):
        self.ideal_similarity = self.ideal_similarity.to(x.device)
        return torch.norm(x - self.ideal_similarity)
