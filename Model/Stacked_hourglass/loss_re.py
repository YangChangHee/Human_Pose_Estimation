#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch

class HeatmapLoss(torch.nn.Module):

    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        L = ((pred - gt)**2)
        L = L.mean(dim=3).mean(dim=2)#.mean(dim=1)
        
        if torch.min((gt.mean(dim=3).mean(dim=2)))==0:
            check_gt=gt.mean(dim=3).mean(dim=2)
            dictn={}
            for n,i in enumerate(check_gt):
                list_0=[]
                for n1,j in enumerate(i):
                    if j==0:
                        list_0.append(n1)
                if list_0 !=[]:
                    dictn[n]=list_0
            dic=dictn.items()
            for i in dic:
                for j in i[1]:
                    L[i[0]][j]=0
            L=L.mean(dim=1)
        else : L=L.mean(dim=1)
        return L

