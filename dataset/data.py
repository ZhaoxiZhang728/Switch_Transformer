# Created by zhaoxizh@unc.edu at 15:50 2023/11/19 using PyCharm
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import FashionMNIST
import lightning as L
import torch
from utils.data_transformer import data_transform,target_transformer
class FMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32,transformer = None,target_transform = None):
        super().__init__()
        self.transfomer = transformer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_transform = target_transform
    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.mnist_test = FashionMNIST(self.data_dir, train=False,transform=self.transfomer,target_transform=self.target_transform)

        mnist_full = FashionMNIST(self.data_dir, train=True,transform=self.transfomer,target_transform=self.target_transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == '__main__':
    datamodule = FMNISTDataModule('E:\pythonproject\pythonProject\VIT\data\FM',batch_size=16,transformer=data_transform,target_transform = target_transformer)
    datamodule.setup('1')
    count = 0
    for batch in datamodule.train_dataloader():
        images, targets = batch
        # `images` variable contains the batch of images
        # `targets` variable contains the corresponding labels/targets
        print(targets)
        print(targets.shape)
        if count ==0:
            break