import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
isz = np.int64(76085)

x1 = torch.load("/Users/taotao/Desktop/本地代码/tiktok/feat_a.pt").type(torch.float32)
x2 = torch.load("/Users/taotao/Desktop/本地代码/tiktok/feat_v.pt").type(torch.float32)
x3 = torch.zeros(isz, 128)
# x3 = torch.load("/Users/taotao/Desktop/本地代码/tiktok/feat_t.pt")
print(f"feat_a = {x1}\tshape = {x1.shape} \nfeat_v = {x2}\tshape={x2.shape}\nfeat_t = {x3}\tshape = {x3.shape}\n")

feature = torch.cat((x2,x1,x3), dim=1)
print(f"after cat = {feature}\tshape = {feature.shape}")



# test = np.load('/Users/taotao/Desktop/本地代码/tiktok/feat_a.pt',encoding = "latin1",allow_pickle=True)  #加载文件
#
# doc = open('what in feat_a.txt', 'a')  #打开一个存储文件，并依次写入

