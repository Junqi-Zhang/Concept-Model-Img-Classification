import copy
import torch

from typing import Any, Dict


def save_model(model: torch.nn.Module, model_path: str) -> None:
    """
    Save the state dictionary of a PyTorch model to the specified path.

    :param model: The PyTorch model to be saved
    :param model_path: The file path where the model will be saved
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}.")


def load_model(model: torch.nn.Module, model_path: str) -> None:
    """
    Load the state dictionary of a PyTorch model from the specified path.

    :param model: The PyTorch model to load the state dictionary into
    :param model_path: The file path containing the model's state dictionary
    """
    model.load_state_dict(torch.load(model_path))
    print(f"Checkpoint loaded from {model_path}.")



class Recorder:
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Recorder object with keyword arguments.
        """
        self.update(kwargs)

    def get(self, name: str, default: str = "none_placeholder") -> Any:
        """
        Get the value of the Recorder item with the given name. 
        If the name does not exist and a default value is provided, return the default value.
        """
        if default == "none_placeholder":
            return self.__dict__[name]
        else:
            return self.__dict__.get(name, default)

    def update(self, update_info: Dict[str, Any]) -> None:
        """
        Update the Recorder with the given dictionary.
        """
        self.__dict__.update(update_info)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Recorder to a dictionary.
        """
        return copy.deepcopy(self.__dict__)
