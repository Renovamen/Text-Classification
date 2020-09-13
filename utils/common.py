import torch
import os

'''
save a model checkpoint

input params:
    epoch: epoch number
    model: model
    model_name: model name
    optimizer: optimizer
    dataset_name: dataset name
    word_map: word2ix map
    checkpoint_path (str): path to save checkpoint
    checkpoint_basename (str): basename of the checkpoint
    best_acc: best accuracy achieved so far (not necessarily in this checkpoint)
    is_best: is this checkpoint the best so far?
    epochs_since_improvement: number of epochs since last improvement
'''
def save_checkpoint(epoch, model, model_name, optimizer, dataset_name, word_map, checkpoint_path, checkpoint_basename = 'checkpoint'):
    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'dataset_name': dataset_name,
        'word_map': word_map
    }
    torch.save(state, os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar'))


'''
load a checkpoint, so that we can continue to train on it

input params:
    checkpoint_path: path to the checkpoint

return ():
    model: /
    model_name: model name
    optimizer: optimizer to update model's weights
    dataset_name: dataset name
    word_map: word2ix map
    start_epoch: we should start training the model from __th epoch
'''
def load_checkpoint(checkpoint_path, device):
    
    checkpoint = torch.load(checkpoint_path, map_location = str(device))
   
    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1
    
    return model, model_name, optimizer, dataset_name, word_map, start_epoch


'''
clip gradients computed during backpropagation to prevent gradient explosion

input params:
    optimizer: optimized with the gradients to be clipped
    grad_clip: gradient clip value
'''
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


'''
keeps track of most recent, average, sum, and count of a metric
'''
class AverageMeter(object):
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


'''
shrinks learning rate by a specified factor

input params:
    optimizer: optimizer whose learning rates must be decayed
    scale_factor: factor in interval (0, 1) to multiply learning rate with
'''
def adjust_learning_rate(optimizer, scale_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))