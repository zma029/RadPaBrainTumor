import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class Test_model(nn.Module):

    def __init__(self, args: dict):

        super().__init__()
        self.args = args
        num_classes = 5
        self.model = DenseNet(num_classes).to(args['device'])

        
        weight_dict = torch.load(args['best_ckpt'], map_location =args['device'])
        self.model.load_state_dict(weight_dict)
        self.model.eval()
        print('Successfully loading checkpoint.')

    def forward(self, images):
        
        prediction = self.model(images)
        
        return prediction