import torch

class ModelConfig:
    # Device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 44