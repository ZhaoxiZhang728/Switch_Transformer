# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import torch
from model.train_model import Moe_model
from dataset.data import FMNISTDataModule
from utils.data_transformer import data_transform
import lightning as L
def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--data_dir',type = str,default='/path/to/load/dataset')
    parser.add_argument('--checkpoint_dir',type=str,default='/path/to/save/checkpoint')
    parser.add_argument('--hidden_dim_patch',type =int,default=32)
    parser.add_argument('--max_epoch',type = int,default=20)
    parser.add_argument('--img_shape',type=tuple,default=(64,64))
    parser.add_argument('--patch_shape',type=tuple,default=(8,8))
    parser.add_argument('--channels',type = int,default=1)
    parser.add_argument('--hidden_dim_expert',type =int, default=64)
    parser.add_argument('--num_class',type= int,default=10)
    parser.add_argument('--num_experts',type=int,default=10)
    parser.add_argument('--optimizer_type',type=str,default='adam')
    parser.add_argument('--lr',type=float,default=0.0001)
    args = parser.parse_args()
    return args

def main():
    args = set_args()
    dataset = FMNISTDataModule(data_dir=args.data_dir,
                               batch_size=args.batch_size,
                               transformer=data_transform)
    model = Moe_model(
        img_size = args.img_shape,
        patch_size = args.patch_shape,
        inchannels=args.channels,
        hidden_dim_patch=args.hidden_dim_patch,
        hidden_dim_expert=args.hidden_dim_expert,
        num_of_class=args.num_class,
        num_experts=args.num_experts,
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        mid_feautre=84,
        alpha=0.5
    )
    if torch.cuda.is_available():
        trainer = L.Trainer(accelerator='gpu',
                            max_epochs=args.max_epoch,
                            check_val_every_n_epoch=1,
                            default_root_dir=args.checkpoint_dir)
    else:
        trainer = L.Trainer(accelerator='cpu',
                            max_epochs=args.max_epoch,
                            check_val_every_n_epoch=1,
                            default_root_dir=args.checkpoint_dir)

    trainer.fit(model=model,datamodule=dataset)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
