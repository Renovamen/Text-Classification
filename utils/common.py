import os
from typing import Tuple, Dict
import torch
from torch import nn, optim

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    model_name: str,
    optimizer: optim.Optimizer,
    dataset_name: str,
    word_map: Dict[str, int],
    checkpoint_path: str,
    checkpoint_basename: str = 'checkpoint'
) -> None:
    """
    Save a model checkpoint

    Parameters
    ----------
    epoch : int
        Epoch number the current checkpoint have been trained for

    model : nn.Module
        Model

    model_name : str
        Name of the model

    optimizer : optim.Optimizer
        Optimizer to update the model's weights

    dataset_name : str
        Name of the dataset

    word_map : Dict[str, int]
        Word2ix map

    checkpoint_path : str
        Path to save the checkpoint

    checkpoint_basename : str
        Basename of the checkpoint
    """
    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'dataset_name': dataset_name,
        'word_map': word_map
    }
    save_path = os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar')
    torch.save(state, save_path)

def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, optim.Optimizer, str, Dict[str, int], int]:
    """
    Load a checkpoint, so that we can continue to train on it

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint to be loaded

    device : torch.device
        Remap the model to which device

    Returns
    -------
    model : nn.Module
        Model

    model_name : str
        Name of the model

    optimizer : optim.Optimizer
        Optimizer to update the model's weights

    dataset_name : str
        Name of the dataset

    word_map : Dict[str, int]
        Word2ix map

    start_epoch : int
        We should start training the model from __th epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1

    return model, model_name, optimizer, dataset_name, word_map, start_epoch

def clip_gradient(optimizer: optim.Optimizer, grad_clip: float) -> None:
    """
    Clip gradients computed during backpropagation to avoid explosion of gradients.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer with the gradients to be clipped

    grad_clip : float
        Gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:
    """
    Keep track of most recent, average, sum, and count of a metric
    """
    def __init__(self, tag = None, writer = None):
        self.writer = writer
        self.tag = tag
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # tensorboard
        if self.writer is not None:
            self.writer.add_scalar(self.tag, val)

def adjust_learning_rate(optimizer: optim.Optimizer, scale_factor: float) -> None:
    """
    Shrink learning rate by a specified factor.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer whose learning rate must be shrunk

    shrink_factor : float
        Factor in interval (0, 1) to multiply learning rate with
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
