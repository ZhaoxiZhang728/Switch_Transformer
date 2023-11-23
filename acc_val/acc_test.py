# Created by zhaoxizh@unc.edu at 18:22 2023/11/20 using PyCharm
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import argparse
import torch
from model.train_model import Moe_model
from utils.data_transformer import data_transform

def dataset(data_dir):
    mnist_test = FashionMNIST(data_dir, train=False,transform=data_transform)
    dl = DataLoader(mnist_test, batch_size=16)
    return dl
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='/path/to/load/dataset')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/path/to/save/dataset')
    parser.add_argument('--hidden_dim_patch', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--img_shape', type=tuple, default=(64, 64))
    parser.add_argument('--patch_shape', type=tuple, default=(8, 8))
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--hidden_dim_expert', type=int, default=64)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--load_ckpt_dir',type=str,default='/path/to/load/checkpoint')
    args = parser.parse_args()
    return args
def check_acc(model,dataset):
    count = 0
    right_predict = 0
    for batch in dataset:
        [img,target] = batch
        output = model(img)

        y_hat = torch.argmax(output,dim=1)

        right_predict += (torch.sum(y_hat == target) / len(img))

        count +=1
    return right_predict / count
def main():
    args = set_args()
    ds = dataset(args.data_dir)

    model = Moe_model.load_from_checkpoint(
        args.load_ckpt_dir,
        img_size=args.img_shape,
        patch_size=args.patch_shape,
        inchannels=args.channels,
        hidden_dim_patch=args.hidden_dim_patch,
        hidden_dim_expert=args.hidden_dim_expert,
        num_of_class=args.num_class,
        num_experts=args.num_experts,
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        mid_feautre=84,
        alpha=0.5)

    model = model.to('cpu')
    accuracy = check_acc(model=model,dataset =ds)

    return accuracy


if __name__ == '__main__':
    acc = main()

    print(acc)