import torch

def save(model, model_path):
    torch.save(model.state_dict(), model_path)
    print('Model saved.')


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))
    print('Best checkpoint loaded.')