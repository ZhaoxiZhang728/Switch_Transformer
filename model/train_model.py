# Created by zhaoxizh@unc.edu at 15:05 2023/11/19 using PyCharm

import lightning as pl
from model.Moe import Moe
from typing import Any
import torch

class Moe_model(pl.LightningModule):
    def __init__(self,
                 inchannels,
                 hidden_dim_patch,
                 hidden_dim_expert,
                 num_of_class,
                 patch_size,
                 img_size,
                 num_experts,
                 alpha,
                 mid_feautre,
                 optimizer_type,
                 lr):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.model = Moe(inchannels,
                    hidden_dim_patch,
                    hidden_dim_expert,
                    num_of_class,
                    patch_size,
                    img_size,
                    num_experts,
                    alpha,
                    mid_feautre
                         )

    def forward(self, x) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('test_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params=self.parameters(),lr = self.lr)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params=self.parameters(),lr = self.lr)
        else:
            raise ValueError('pls, input adam or sgd')
        return optimizer

