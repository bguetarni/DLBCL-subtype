import torch
import torch.nn as nn

def model_parameters_count(model):
    """
    Count the model total number of parameters

    args:
        model : (nn.Module)

    return a dict with name of module and number of params
    """
    trainable, not_trainable = 0, 0
    for p in model.parameters():
        count = p.flatten().size()[0]
        if p.requires_grad:
            trainable += count
        else:
            not_trainable += count
    
    return dict(trainable=trainable, fixed=not_trainable)


def smooth_cross_entropy(pred, targets, p=0.0, n_classes=None):
    """
        args:
            pred : model output (torch.Tensor)
            targets : ground-truth as index (torch.Tensor)
            p : probability for label smoothing (float)
            n_classes : number of classes (int)

        return tensor of loss (torch.Tensor)
    """
    
    def smooth_labels_randomly(targets, p, n_classes):
        # produce on-hot encoded vector of targets
        # fill true classes value with random value in : [1 - p, 1]
        # and completes the other to sum up to 1
        
        res = torch.zeros((targets.size(0), n_classes), device=targets.device)
        rand = 1 - torch.rand(targets.data.unsqueeze(1).shape, device=targets.device)*p
        res.scatter_(1, targets.data.unsqueeze(1), rand)
        fill_ = (1 - res.sum(-1))/(n_classes - 1)
        return res.maximum(fill_.unsqueeze(dim=-1).repeat(1, n_classes))

    assert isinstance(pred, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    
    if p:
        if n_classes is None:
            n_classes = pred.size(-1)
            
        targets = smooth_labels_randomly(targets, p, n_classes)
        pred = pred.log_softmax(dim=-1)
        cce_loss = torch.sum(-targets * pred, dim=1)
        return torch.mean(cce_loss)
    else:
        return nn.functional.cross_entropy(pred, targets.to(dtype=torch.long))
