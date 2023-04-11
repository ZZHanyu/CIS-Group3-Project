
# Group3-Project

## 3/23 Zhy:

>1.  更新部分注释，在代码中标记论文中出现的数学公式

## 3/26 Zhy:

>1.  使用A100显卡跑完了训练数据，更新weight和log文件数据（weight代表训练结束后的权重）
>2.  更新使用训练出来的参数跑 InvRL 
>3.  更新部分注释，和标注写 ERM 方法的位置


## 4/09 Zhy：

>1.  压缩数据集至 1% 规模
>### 调整数据集规模教程：
>>Step 1: 首先打开 ./tiktop 目录下的 "load_data.py" 文件，找到代码第 33 行，如下图：  
>>![image](https://github.com/ZZHanyu/Group-3-Project/blob/main/IMAG/image1.png)  
>>改 / 100 这个数字，除以几相当于数据集规模缩减至几分之一  
>>Step 2:其次，打开InvRL.py文件，第131行和第175行，根据Step1进行相应修改（Step1 /100，则这一步也是/100），如下图:  
>>![image](https://github.com/ZZHanyu/Group-3-Project/blob/main/IMAG/image3.png)  
>>![image](https://github.com/ZZHanyu/Group-3-Project/blob/main/IMAG/image4.png)  
>>Step 3: 若出现电脑卡顿，蓝屏等问题，可能是batch size 设置较大，需要移步 main.py 文件，找到第 32 行，如下图:  
>>![image](https://github.com/ZZHanyu/Group-3-Project/blob/main/IMAG/image2.png)  
>>改"default = 512"后面的数字，没有显卡的电脑推荐小于32；有显卡可以稍微大一点，设64/128之间（取决于显存大小，如果跑不成功就尽量设置小一点）  
>>（Tips：通过调整这两个数字的大小可以调整训练时间，需要根据不同电脑不同调整，请大家试试～）  


## 4/11 Zhy:

>1. 运行完结果体现在 更新在tiktok中的xxx.npy数据文件，以及weights、logs 权重、日志文件
>2. 其中log文件记录程序的输出，其中，nDCG值越接近1越精准  
### ✨近期目标🌟:  
>(1) ERM和Attention model的代码，可以分开写在不同的.py文件中  
