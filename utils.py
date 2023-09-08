import torch


###################################
# model operations
###################################

def save(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}.")


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Best checkpoint loaded from {model_path}.")


###################################
# loss functions
###################################

def capped_lp_norm(x, p=0.5, gamma=0.1, epsilon=1e-7):
    lp = torch.pow(torch.abs(x) + epsilon, p)

    loss = torch.min(lp, gamma ** p * torch.ones_like(x))
    return loss.sum(dim=1).mean()


def entropy(x, epsilon=1e-7):
    return -(torch.abs(x) * torch.log(torch.abs(x) + epsilon)).sum(dim=1).mean()


def smoothly_clipped_absolute_deviation(x, lmbda=0.1, gamma=2):
    x_abs = torch.abs(x)

    loss_1 = lmbda * x_abs
    loss_2 = (2 * gamma * lmbda * x_abs - x **
              2 - lmbda ** 2) / (2 * (gamma - 1))
    loss_3 = (gamma + 1) * lmbda ** 2 / 2

    loss = torch.where(x_abs < lmbda, loss_1,
                       torch.where(x_abs < gamma * lmbda, loss_2, loss_3))

    return loss.sum(dim=1).mean()


def orthogonality_l2_norm(x):
    ideal_similarity = torch.eye(
        x.size(0),
        dtype=torch.float,
        device=x.device
    )
    return torch.norm(x-ideal_similarity)


###################################
# PID Controllers
###################################

class PIController(object):
    def __init__(self, kp, ki, target_metric, forgetting_factor=0.95, initial_weight=1.0):
        """
        ...
        :param forgetting_factor: Value between 0 and 1 to make old errors decay over time.
        """
        self.kp = kp
        self.ki = ki
        self.target_metric = target_metric
        self.forgetting_factor = forgetting_factor
        self.initial_weight = initial_weight
        self.weight = initial_weight
        self.cumulative_error = 0.0

    def update(self, current_metric):
        error = current_metric - self.target_metric

        # Applying the forgetting factor
        self.cumulative_error = self.forgetting_factor * self.cumulative_error + error

        change = self.kp * error + self.ki * self.cumulative_error
        self.weight = self.initial_weight + change
        self.weight = min(max(self.weight, 0.0), 0.2)

        return self.weight
