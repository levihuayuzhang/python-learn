import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(
    root="dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"length of train dataset is: {train_data_size}")
print(f"length of test dataset is: {test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

ass = Ass()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(ass.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("logs")

for i in range(epoch):
    print(f"------{i + 1} epoch starting------")

    ass.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = ass(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print(f"time of training: {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    ass.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = ass(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

    print(f"Loss on total test dataset: {total_test_loss}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(ass, f"ass_{i}.pth")
    print(f"model saved for epoch {i}")

writer.close()
