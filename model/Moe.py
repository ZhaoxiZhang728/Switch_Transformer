# Created by zhaoxizh@unc.edu at 19:38 2023/11/21 using PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
class Patch_embedding(nn.Module):
    def __init__(self,inchannels,hidden_channels,patch_size): # hidden dim for patch embeding
        super().__init__()

        self.conv = nn.Conv2d(in_channels=inchannels,
                              out_channels=hidden_channels,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self,x):
        # [batch,num of patches,num_features]
        return torch.flatten(self.conv(x),start_dim=2,end_dim=3).transpose(1,2)


class Moe_single_expert(nn.Module):
    def __init__(self,inchannel,hidden_channels):# in channel is patch_hidden_dim, hidden dim is for
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=hidden_channels,
                      kernel_size=(3,3),
                      stride=1,
                      padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=(3, 3),
                      stride=1,
                      padding='same'),
            nn.GELU()
        )
    def forward(self,x):
        return self.model(x)
class Router(nn.Module):
    def __init__(self,dim,num_experts,alpha,num_patch):
        super().__init__()
        in_channel = [dim,128,64]
        out_channel = [128,64,32]
        layers = []
        self.dropout = nn.Dropout(alpha)
        self.activation = nn.GELU()
        for i,o in zip(in_channel,out_channel):
            layers += [nn.Conv2d(in_channels=i,
                                 out_channels=o,
                                 kernel_size=(3,3),
                                 stride=1,
                                 padding='same'),
                       self.activation,
                       self.dropout]
        self.layers = nn.Sequential(*layers)
        dense_dim =int(32 * num_patch[0] * num_patch[1])
        self.dense1 = nn.Linear(in_features=dense_dim,out_features=int(dense_dim  / 2))

        self.dense2 = nn.Linear(in_features=int(dense_dim  / 2),out_features=num_experts)

    def forward(self,x):
        output = self.layers(x)
        output = torch.flatten(output,start_dim=1,end_dim=3)
        output = self.dropout(self.activation(self.dense1(output)))

        output = self.dense2(output)

        return output,torch.argmax(output,dim=1)

class Self_Attention(nn.Module):

    def __init__(self, num_input, num_output,num_patch):
        super().__init__()
        self.num_patch = num_patch
        # for output of attention
        self.weight_o = nn.Linear(num_input, num_output, bias=True)

    def forward(self, queries, keys, values, valid_lens):

        output = self.attention(queries, keys, values, valid_lens)

        return self.weight_o((output))

    def attention(self, q, k, v, valid_lens):
        k = torch.transpose(input=k, dim0=1, dim1=2)
        attenion_score = torch.bmm(q, k) / torch.sqrt(valid_lens)

        attenion_weight = F.softmax(attenion_score, dim=-1)

        return torch.bmm(attenion_weight, v)




class Moe(nn.Module):
    def __init__(self,
                 inchannels,
                 hidden_dim_patch,
                 hidden_dim_expert,
                 num_of_class,
                 patch_size,
                 img_size,
                 num_experts,
                 alpha,
                 mid_feautre
                 ):
        super().__init__()
        h,w = img_size
        self.num_patch = (h//patch_size[0],w//patch_size[1])
        self.patch_embedding = Patch_embedding(inchannels = inchannels,
                                               hidden_channels = hidden_dim_patch,
                                               patch_size = patch_size)

        self.self_attention = Self_Attention(hidden_dim_patch,hidden_dim_patch,self.num_patch)
        self.norm_patch = nn.BatchNorm2d(hidden_dim_patch)
        self.norm_output = nn.BatchNorm2d(hidden_dim_expert)
        self.pos_embedding = nn.Parameter(torch.randn(1,self.num_patch[0] * self.num_patch[1],hidden_dim_patch))

        self._1x1_shard_layer = nn.Conv2d(in_channels=hidden_dim_patch,out_channels=hidden_dim_expert,stride=1,kernel_size=1)
        experts = []
        for _ in range(num_experts):
            experts.append(Moe_single_expert(inchannel=hidden_dim_patch,
                                             hidden_channels = hidden_dim_expert))

        self.router = Router(
                             num_patch=self.num_patch,
                             dim = hidden_dim_patch,
                             num_experts = num_experts,
                             alpha = alpha)
        self.experts = nn.Sequential(*experts)

        self.output_layers = nn.Sequential(
            nn.Linear(in_features=hidden_dim_expert * self.num_patch[0] * self.num_patch[1],
                      out_features=mid_feautre),
            nn.Dropout(alpha),
            nn.GELU(),
            nn.Linear(in_features=mid_feautre,out_features=num_of_class)
        )

    def forward(self,x):
        y = self.patch_embedding(x)

        y = y + self.pos_embedding

        re = self.self_attention(y,y,y,torch.tensor(self.num_patch[0]))
        y = self.transpose_output(y)
        re = self.transpose_output(re)
        out = self.norm_patch(y + re)

        posb,poses_expert = self.router(out)
        result = []
        for i in range(len(poses_expert)):
            pos = poses_expert[i]
            result.append(posb[i][pos] * self.experts[pos](re[i]))

        re = torch.stack(result)
        out = self._1x1_shard_layer(out)

        output = self.norm_output(re + out)

        output = torch.flatten(output,start_dim=1,end_dim=3)
        output = self.output_layers(output)

        return output

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.permute(0, 2, 1)
        # [batch_size,number of patch, number of features]
        return X.reshape(X.shape[0], X.shape[1], self.num_patch[0], self.num_patch[1])
if __name__ == '__main__':
    img = torch.rand((16, 3, 32, 32))
    moe = Moe(inchannels =3,
                 hidden_dim_patch = 32,
                 hidden_dim_expert = 64,
                 num_of_class =10,
                 patch_size =(8,8),
                 img_size = (32,32),
                 num_experts = 10,
                 alpha = 0.5,
                 mid_feautre = 84)
    re = moe(img)
    print(re.shape)











