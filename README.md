<<<<<<< HEAD
# Invariant Representation Learning for Multimedia Recommendation

This our official implementation via Pytorch for the paper:

>Xiaoyu Du, Zike Wu, Fuli Feng, Xiangnan He and Jinhui Tang. Invariant Representation Learning for Multimedia Recommendation. In ACM MM`22, October 10–14, 2022, Lisboa, Portugal.

# Requirements

* Python==3.8.10
* Pytorch==1.11.0+cu113
* numpy, scipy, argparse, logging, sklearn, tqdm

# Example to Run the Codes
First, we run UltraGCN for pretraining：
```bash
python main.py --model UltraGCN
```
After that, we run the InvRL model:
```bash
python main.py --model InvRL
```
=======
# Group3-Project

## 3.23 Zhy:

1.  更新部分注释，在代码中标记论文中出现的数学公式

## 3.26 Zhy:

1.  使用A100显卡跑完了训练数据，更新weight和log文件数据（weight代表训练结束后的权重）
2.  更新使用训练出来的参数跑 InvRL 
3.  更新部分注释，和标注写 ERM 方法的位置


>>>>>>> 9eecbbc3cee827b408aa3fbb16f94ab7ecd58151
