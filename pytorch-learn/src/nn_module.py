import torch
from torch import nn


class Ass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


ass = Ass()
x = torch.tensor(1.0)
output = ass(x)
print(output)
