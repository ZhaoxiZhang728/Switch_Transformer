# Created by zhaoxizh@unc.edu at 19:38 2023/11/21 using PyCharm
# Created by zhaoxizh@unc.edu at 16:32 2023/11/19 using PyCharm

from torchvision import transforms
import torch
import torch.nn.functional as F

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64)),
    transforms.Normalize((0.5,), (0.5,))
])

target_transformer = transforms.Compose([
    lambda x: torch.LongTensor([x]),  # or just torch.tensor
    lambda x: F.one_hot(x, 10),
    lambda x: torch.squeeze(x)]
)