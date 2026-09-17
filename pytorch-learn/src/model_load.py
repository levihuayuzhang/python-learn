import torch
import torchvision

# from torch import nn
from model_save import *

model = torch.load("vgg16_method1.pth", weights_only=False)
# print(model)


vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth", weights_only=False))
# model = torch.load("vgg16_method2.pth", weights_only=False)
# print(vgg16)


# class Ass(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x


model = torch.load("ass_method1.pth", weights_only=False)
print(model)
