import torch.nn as nn

class ImgClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10,output_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        y = self.layers(x)
        
        return y