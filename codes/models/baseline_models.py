
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from prettytable import PrettyTable

# from multibox_loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6),
                 stride=(1, 1), padding=(5, 5), pool_size=None):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=True)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=None, pool_type='max'):

        if pool_size is None: pool_size = self.pool_size


        x = self.conv(x) # F.relu_(self.conv(input1))
        if self.pool_size is None:return x
        #
        #x = self.bn2(self.conv2(x))
        if pool_type == 'max':
             x = F.max_pool2d(x, kernel_size=pool_size)
        else :x = F.avg_pool2d(x, kernel_size=pool_size)
        #print(torch.sum(x))
        #print(torch.sum(self.conv.weight))
        return x


class FeatureExtractor_baseline(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10],drop_out=0,mode_train=0):
        super().__init__()
        self.num_blocks = len(channels)
        self.start_channel = start_channel
        self.l = nn.ModuleList()
        self.input_image_dim = tuple(input_image_dim)
        self.fc1_p = fc1_p
        self.mode_train = mode_train
        self.a = torch.nn.ReLU()
        self.a2=torch.nn.Softmax(dim=1)
        self.norms=nn.ModuleList()



        self.dropout = nn.Dropout(drop_out)

        last_channel = start_channel
        for i in range(self.num_blocks):
            self.l.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]
            self.norms.append(torch.nn.BatchNorm2d(channels[i]))
            #self.norms.append(lambda x:x)

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()


        self.l.append( nn.Linear(conv_dim[0], fc1_p[0], bias=True))
        self.norms.append(torch.nn.BatchNorm1d(fc1_p[0]))
        self.l.append(nn.Linear(fc1_p[0], fc1_p[1], bias=True))
        self.num_layers = self.num_blocks + 2
        self.init_weight()

        count_parameters(self)

    def get_conv_output_dim(self):
        input_ = torch.Tensor(np.zeros((1, self.start_channel) + self.input_image_dim))
        x = self.cnn_feature_extractor(input_)
        print(x.shape)
        return len(x.flatten()), x.shape

    @staticmethod
    def init_layer(layer):
        if str(type(layer)).find('ConvBlock')>=0:
            layer=layer.conv
            #nn.init.uniform_(layer.weight,a=-1,b=1 )
            nn.init.kaiming_uniform_(layer.weight)
            #nn.init.xavier_uniform_(layer.weight)
        else:
            #nn.init.uniform_(layer.weight,a=-1,b=1 )
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_layers):
            self.init_layer(self.l[i])

    def cnn_feature_extractor(self, x,layer=None):
        # input 501*64
        if layer is not None:max_layer=min(self.num_blocks,layer+1)
        else :max_layer=self.num_blocks
        for i in range(max_layer):
            x = self.l[i](x)
            x = self.a(x)
            #if i == 0: print(torch.sum(self.norms[i].running_mean))
            #x=self.norms[i](x)

            if self.mode_train == 1:
                x = self.dropout(x)
            else :
                x = 0.9*x


        return x
    def pre_processing(self,x):
        return x
    def forward(self, x):
        x=self.pre_processing(x)
        x = self.cnn_feature_extractor(x)

        x=x.flatten(start_dim=1)
        x = self.a(self.l[self.num_layers-2](x))
        if self.mode_train == 1:
            x = self.dropout(x)
        else:
            x = 0.9 * x
        #x=self.norms[-1](x)
        x = self.l[self.num_layers - 1](x)
        x=self.post_processing(x)
        return x

    def post_processing(self, x):
        return self.a2(x)
    def run_till(self,x,layer):
        x = self.pre_processing(x)
        x = self.cnn_feature_extractor(x,layer)
        if layer<self.num_blocks:return x
        x = x.flatten(start_dim=1)
        if layer>(self.num_layers-2):
            x = self.a(self.l[self.num_layers - 2](x))
            x = self.norms[-1](x)
        if layer > (self.num_layers - 1):
            x = self.l[self.num_layers - 1](x)

        return x




class ResidualBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2,stride=1,padding=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self,blocks, channels,start_channel, input_image_dim,convs,pads,strides,fc1_p,mode_train=0,drop_out=0.1):
        super(ResNet, self).__init__()
        self.mode_train=mode_train
        self.drop_out=drop_out
        self.dropout = nn.Dropout(drop_out)
        self.blocks=blocks
        self.num_blocks=len(self.blocks)
        self.input_image_dim=input_image_dim
        self.a = torch.nn.ReLU()
        self.a2 = torch.nn.Softmax(dim=1)
        self.l= nn.ModuleList()
        self.norms=nn.ModuleList()
        last_channel=start_channel
        i=0
        self.l.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                        kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                         padding=pads[i]))
        last_channel = channels[i]
        for i in range(1,self.num_blocks):
            for j in range(blocks[i]):
                self.norms.append(torch.nn.BatchNorm2d(last_channel))
                self.l.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                        kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                         padding=pads[i]))

                self.l.append(ConvBlock(in_channels=channels[i], out_channels=channels[i],
                                        kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                        padding=pads[i]))
                self.norms.append(torch.nn.BatchNorm2d(channels[i]))
                last_channel=channels[i]


        self.avgpool = nn.AvgPool2d(kernel_size=input_image_dim[0], stride=1)

        self.norms.append(torch.nn.BatchNorm2d(last_channel))
        self.l.append(nn.Linear(last_channel, fc1_p[1], bias=True))
        self.num_layers = len(self.l)
        #self.l.append( nn.Linear(fc1_p[0], fc1_p[1]))
        self.init_weight()
        count_parameters(self)




    def forward(self, x):

        x=self.pre_processing(x)
        x=self.l[0](x) #first conn layer
        loop=1
        for i in range(1,self.num_blocks):
            for j in range(self.blocks[i]):#source https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
                residual =x
                x=self.norms[loop-1](x)
                x=self.a(x)
                x = self.l[loop](x) #input_dim*128channels->input_dim*channels[i]
                x = self.norms[loop ](x)
                x=self.a(x)
                x = self.a(self.l[loop+1](x))

                if x.shape[1]!=residual.shape[1]:residual=torch.cat([residual,torch.zeros(residual.shape)],dim=1)
                #print(x[0, 0, 16, 16] )
                x=x+residual
                loop+=2
                # if self.mode_train==1:x=self.dropout(x)
                # else:x=(1-self.drop_out)*x
        x=self.norms[loop-1](x)
        x=self.a(x)
        x=self.avgpool(x).flatten(start_dim=1)
        if self.mode_train == 1:
            x = self.dropout(x)
        else:
            x = (1 - self.drop_out) * x
        x=self.l[loop](x)
        x=self.post_processing(x)
        return x

    def post_processing(self, x):
        return self.a2(x)

    def pre_processing(self, x):
        return x

    def run_till(self,x,layer):
        x = self.pre_processing(x)
        x = self.l[0](x)  # first conn layer
        loop = 1
        for i in range(1, self.num_blocks):
            for j in range(self.blocks[
                               i]):  # source https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
                residual = x
                x = self.norms[loop - 1](x)
                x = self.a(x)
                x = self.l[loop](x)  # input_dim*128channels->input_dim*channels[i]
                x = self.norms[loop](x)
                x = self.a(x)
                x = self.a(self.l[loop + 1](x))

                if x.shape[1] != residual.shape[1]: residual = torch.cat([residual, torch.zeros(residual.shape)], dim=1)
                # print(x[0, 0, 16, 16] )
                x = x + residual
                loop += 2


        x = self.avgpool(x).flatten(start_dim=1)
        x = self.l[loop](x)
        return x

    @staticmethod
    def init_layer(layer):
        if str(type(layer)).find('ConvBlock') >= 0:
            layer = layer.conv
            #nn.init.uniform_(layer.weight,a=,b=1 )
            nn.init.kaiming_uniform_(layer.weight)
            # nn.init.xavier_uniform_(layer.weight)
            #layer.weight.data.fill_(0.01)
        else:
            # nn.init.uniform_(layer.weight,a=-1,b=1 )
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_uniform_(layer.weight)
            #layer.weight.data.fill_(1)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_layers):
            self.init_layer(self.l[i])
class fc_model(nn.Module):
    def __init__(self,input_node,layers):
        super().__init__()
        self.l=nn.ModuleList()
        for i in range(len(layers)):
            self.l.append(nn.Linear(input_node,layers[i], bias=True))
            input_node=layers[i]
        self.num_layers=i+1
        self.a=torch.nn.ReLU()

        self.a2 = torch.nn.Softmax(dim=1)
        self.init_weight()
        count_parameters(self)
    def post_processing(self,x):
        return self.a2(x)
    def pre_processing(self,x):
        return x.flatten(start_dim=1) #/256
    def forward(self,x):
        x=self.run_till(x,self.num_layers)
        x=self.post_processing(x)
        return x
    def run_till(self,x,layer):
        x=self.pre_processing(x)
        for i in range(layer):
            x=self.l[i](x)
            x=self.a(x)
        return x
    def init_layer(self,layer):
        nn.init.xavier_uniform_(layer.weight )
        #nn.init.uniform_(layer.weight,-1,1)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_layers):
            self.init_layer(self.l[i])
        #nn.init.uniform_(self.l[0].weight, -100, 100)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return table,total_params
if __name__ == '__main__':

    from codes.train.losses import custom_EntropyLoss
    from types import SimpleNamespace
    import yaml
    with open('../train/config.yaml', 'r') as f:
            config = SimpleNamespace(**yaml.safe_load(f))
    #d=grad_information()
    model=ResNet(**config.model_paramsrn1)
    d=model(torch.ones((1,3,32,32)))
    c=0
    # y_hat=model(torch.ones(1,32,32,3))
    # loss_f=custom_EntropyLoss()
    # loss = loss_f(y_hat, torch.tensor([1]))
    # loss.backward()
    # grads = d.get_grads(model)
    #
    # k=0
