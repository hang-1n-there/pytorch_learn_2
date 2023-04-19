import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,(3,3),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels,out_channels,(3,3),stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self,x):
        # |x| = (Batch_size, in_channel, H , W)
        y = self.layers(x)
        # |y| = (Batch_size, out_channel, H , W)