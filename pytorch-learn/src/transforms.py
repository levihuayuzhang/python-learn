from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "pytorch-learn/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("pytorch-learn/logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
