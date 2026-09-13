import torchvision

# train_data = torchvision.datasets.ImageNet(
#     "dataset", split="train", download=True, transform=torchvision.transforms.ToTensor()
# )

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(
    "dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True
)

vgg16_true.add_module(
    "add_linear",
)
