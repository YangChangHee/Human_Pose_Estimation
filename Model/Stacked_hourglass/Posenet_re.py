#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
from Basic_Hourglass import Conv, Hourglass, Pool, Residual
from loss_re import HeatmapLoss

class UnFlatten(nn.Module):
    def forward(self,input):
        return input.view(-1,256,4,4)

class Merge(nn.Module):
    def __init__(self,x_dim,y_dim):
        super(Merge,self).__init__()
        self.conv=Conv(x_dim,y_dim,1,relu=False,bn=False)
    def forward(self,x):
        return self.conv(x)

class PoseNet(nn.Module):
    def __init__(self,nstack,inp_dim,out_dim,bn=False,increase=0,**kwargs):
        super(PoseNet,self).__init__()
        
        self.nstack=nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 3, 1, bn=True, relu=True), # kernel 7 stride 2 padiing 3
            Residual(64, 128),
            #Pool(2, 2), #=> 원래 있었음.
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        ################## nstack => hourglass, features Num #####################
        self.hgs=nn.ModuleList([
            nn.Sequential(
            Hourglass(4,inp_dim,bn,increase),
            ) for i in range(nstack)])

        self.features=nn.ModuleList([
            nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True,relu=True))
            for i in range(nstack)])
        
        self.outs=nn.ModuleList([Conv(inp_dim, out_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features=nn.ModuleList([Merge(inp_dim,inp_dim) for i in range(nstack-1)])
        self.merge_preds =nn.ModuleList([Merge(out_dim,inp_dim) for i in range(nstack-1)])
        self.nstack =nstack
        self.heatmapLoss=HeatmapLoss()
        
    def forward(self,imgs):
        x= imgs.permute(0,3,1,2) # => image 재배치
        x= self.pre(x) #=> inp_dim 으로 바꾸기 위한 첫 작업
        combined_hm_preds=[]
        for i in range(self.nstack):
            hg=self.hgs[i](x)
            feature=self.features[i](hg)
            preds=self.outs[i](feature)
            print(preds.shape)
            combined_hm_preds.append(preds)
            if i < self.nstack -1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
                
        return torch.stack(combined_hm_preds,1)
    
    def calc_loss(self,combined_hm_preds,heatmaps):
        combined_loss=[]
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:,i],heatmaps)) #=> combined_hm_preds[0][:,i]에서 [:,i]로 변경
        combined_loss=torch.stack(combined_loss, dim=1)
        return combined_loss


# In[ ]:




