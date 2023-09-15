import torch
from torch import Tensor


def capped_lp_norm(x: Tensor,
                   p: float = 0.5, gamma: float = 0.1,
                   epsilon: float = 1e-7, reduction: str = "sum") -> Tensor:
    """
    Calculate the capped Lp norm of the input tensor x.

    Args:
        x (Tensor): Input tensor.
        p (float, optional): The power for the Lp norm. Default is 0.5.
        gamma (float, optional): The upper bound for the capped Lp norm. Default is 0.1.
        epsilon (float, optional): A small constant added to the absolute value of x for numerical stability. Default is 1e-7.
        reduction (str, optional): The reduction method to apply to the loss. Options are "sum" or "mean". Default is "sum".

    Returns:
        Tensor: The capped Lp norm of the input tensor x.
    """
    lp = torch.pow(torch.abs(x) + epsilon, p)
    loss = torch.clamp(lp, max=gamma ** p)
    reduced_loss = {
        "sum": loss.sum(dim=1),
        "mean": loss.mean(dim=1)
    }
    return reduced_loss[reduction].mean()


def capped_lp_norm_hinge(x: Tensor, target: int = 30, threshold: float = 1e-7,
                         p: float = 0.5, gamma: float = 0.1,
                         epsilon: float = 1e-7, reduction: str = "sum") -> Tensor:
    """
    Calculate the capped Lp norm hinge loss for the input tensor x.

    Args:
        x (Tensor): Input tensor.
        target (int, optional): The target number of elements greater than the threshold. Default is 30.
        threshold (float, optional): The threshold for non-zero elements. Default is 1e-7.
        p (float, optional): The power for the Lp norm. Default is 0.5.
        gamma (float, optional): The upper bound for the capped Lp norm. Default is 0.1.
        epsilon (float, optional): A small constant added to the absolute value of x for numerical stability. Default is 1e-7.
        reduction (str, optional): The reduction method to apply to the loss. Options are "sum" or "mean". Default is "sum".

    Returns:
        Tensor: The capped Lp norm hinge loss for the input tensor x.
    """
    with torch.no_grad():
        switch = (torch.sum(x > threshold, dim=1) > target).type(x.dtype)

    lp = torch.pow(torch.abs(x) + epsilon, p)
    loss = torch.clamp(lp, max=gamma ** p)
    reduced_loss = {
        "sum": loss.sum(dim=1),
        "mean": loss.mean(dim=1)
    }
    return (switch * reduced_loss[reduction]).mean()


def smoothly_clipped_absolute_deviation(x: Tensor, lambda_val: float = 0.1, gamma: float = 2) -> Tensor:
    """
    Calculate the Smoothly Clipped Absolute Deviation (SCAD) loss for the input tensor x.

    Args:
        x (Tensor): The input tensor.
        lambda_val (float, optional): The lambda parameter. Defaults to 0.1.
        gamma (float, optional): The gamma parameter. Defaults to 2.

    Returns:
        Tensor: The SCAD loss.
    """
    x_abs = torch.abs(x)

    loss_1 = lambda_val * x_abs
    loss_2 = (2 * gamma * lambda_val * x_abs - x **
              2 - lambda_val ** 2) / (2 * (gamma - 1))
    loss_3 = (gamma + 1) * lambda_val ** 2 / 2

    loss = torch.where(x_abs < lambda_val, loss_1,
                       torch.where(x_abs < gamma * lambda_val, loss_2, loss_3))

    return loss.sum(dim=1).mean()


def entropy(x: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Calculate the entropy of the input tensor x.

    Args:
        x (Tensor): The input tensor.
        epsilon (float, optional): A small constant to ensure the argument of the logarithm is non-zero. Default is 1e-7.

    Returns:
        Tensor: The entropy of the input tensor x.
    """
    return -(torch.abs(x) * torch.log(torch.abs(x) + epsilon)).sum(dim=1).mean()


def orthogonality_l2_norm(x: Tensor) -> Tensor:
    """
    Calculate the L2 norm between the input tensor x and the ideal similarity matrix (identity matrix).

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The L2 norm between the input tensor x and the ideal similarity matrix.
    """
    ideal_similarity = torch.eye(
        x.size(0),
        dtype=torch.float,
        device=x.device
    )
    return torch.norm(x - ideal_similarity)


class PIController:
    """
    A Proportional-Integral (PI) controller that adjusts a system's response to reach a desired target metric.
    """

    def __init__(self, kp: float, ki: float, target_metric: float, initial_weight: float = 0.0,
                 forgetting_factor: float = 0.95, min_weight: float = 0.0, max_weight: float = 0.2):
        """
        Initialize the PIController instance with the given parameters.

        :param kp: Proportional gain, controls the speed of the system's response to error.
        :param ki: Integral gain, eliminates the system's steady-state error.
        :param target_metric: The desired target metric value.
        :param initial_weight: The initial weight of the controller (default: 0.0).
        :param forgetting_factor: The forgetting factor for calculating the cumulative error (default: 0.95).
        :param min_weight: The minimum value of the weight (default: 0.0).
        :param max_weight: The maximum value of the weight (default: 0.2).
        """
        self.kp = kp
        self.ki = ki
        self.target_metric = target_metric
        self.initial_weight = initial_weight
        self.forgetting_factor = forgetting_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cumulative_error = 0.0

    def update(self, current_metric: float) -> float:
        """
        Update the controller's weight based on the current metric value.

        :param current_metric: The current metric value.
        :return: The updated weight.
        """
        error = current_metric - self.target_metric

        self.cumulative_error = self.forgetting_factor * self.cumulative_error + error

        change = self.kp * error + self.ki * self.cumulative_error
        weight = self.initial_weight + change
        weight = min(max(weight, self.min_weight), self.max_weight)

        return weight
