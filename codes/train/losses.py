import torch.nn as nn
import numpy as np
import torch
EPSILON_FP16 = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class custom_L1Loss(nn.Module):
    def __init__(self,loss_func=nn.L1Loss()):

        super().__init__()



    def forward(self, pred, actual):
        #pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        #if torch.where(torch.abs(actual)<0.0001 :loss=abs(pred)
        loss= torch.abs(pred-actual)/(torch.abs(actual)+0.01)

        return loss.mean()

class custom_EntropyLoss(nn.Module):
    def __init__(self,loss_func=nn.L1Loss()):

        super().__init__()
        self.func=torch.nn.NLLLoss()



    def forward(self, pred, actual):
        #pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        #if torch.where(torch.abs(actual)<0.0001 :loss=abs(pred)
        pred=torch.clamp(pred,0.0001,0.9999)
        loss= self.func(torch.log(pred),actual)


        return loss