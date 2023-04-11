import numpy as np

np.set_printoptions(threshold=np.inf)

test = np.load('/Users/taotao/Documents/GitHub/Group3-Project/weights/tiktok_UGCN_best.pth',encoding = "latin1",allow_pickle=True)  #加载文件

doc = open('what in tiktok_UGCN_best.txt', 'a')  #打开一个存储文件，并依次写入

print(test, file=doc)  #将打印内容写入文件中
