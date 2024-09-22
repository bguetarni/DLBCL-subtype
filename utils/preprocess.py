import torch

def normalize(x, type=None):
    """
    Return normalized torch.Tensor in [-1,1]

    Args:
        x : tensor to normalize (torch.Tensor)
    """

    if type == 'imagenet':
        mean_, std_ = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif type == 'standard':
        mean_, std_ = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_, std_ = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    mean_, std_ = torch.tensor(mean_, dtype=torch.float32), torch.tensor(std_, dtype=torch.float32)
    return (x / 255 - mean_) / std_
