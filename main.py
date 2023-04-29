#yangqiok
import argparse
import sys
import logging
from tiktok.load_data import dataset as ds_tiktok
import numpy as np
import torch
import MultAttention as MA

from UltraGCN import UltraGCN
from InvRL import InvRL
from UltraGCN_ERM import ERMNet
import time
from tqdm import tqdm



def parse_args():
    # argsparse是python的命令行解析的标准模块，内置于python，不需要安装。这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。
    parser = argparse.ArgumentParser(description='Arguments for models')
    parser.add_argument('--dataset', nargs='?', default='tiktok',
                        help='dataset')
    # 这里可以输入多个数据集，文件夹里的数据集只有tiktok，这里我们只用tiktok，若有其他数据集，我们可以 --dataset tiktop kuaishou douyu etc...
    parser.add_argument('--model', nargs='?', default='UltraGCN',
                        help='model type')
    parser.add_argument('--p_emb', nargs='?', default='[0.001,0.0001]',
                        help='lr and reg')
    # 学习率和正则项
    parser.add_argument('--p_proj', nargs='?', default='[0.001,0.01]',
                       help='lr and reg for W')
    # 参数的学习率
    parser.add_argument('--p_embp', nargs='?', default='[0.001,0.0001]',
                        help='lr and reg in predictor')
    # 预测器的学习率
    parser.add_argument('--p_ctx', nargs='?', default='[0.001,0.1]',
                        help='lr and reg for W in predictor')
    parser.add_argument('--p_w', nargs='?', default='[1,1,1,1]',
                        help='w1, w2, w3, w4')
    parser.add_argument('--feat_dim', type=int, default=64,
                        help='feature dim')
    # 特征维度数
    parser.add_argument('--tolog', type=int, default=1,
                        help='0: output to stdout, 1: output to logfile')
    # 这个参数是负责输出到哪的，如果 --tolog=0则输出至你的屏幕，默认输出在logfile
    parser.add_argument('--bsz', type=int, default=512,
                        help='batch size')
    # 批量大小
    parser.add_argument('--ssz', type=int, default=512,
                        help='size of test samples, including positive and negative samples')
    # 测试集大小
    parser.add_argument('--neg_num', type=int, default=50,
                        help='negative samples each batch')
    parser.add_argument('--neighbor_num', type=int, default=10,
                        help='number of item neighbors')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='number of domains')
    parser.add_argument('--regi', type=float, default=0.0,
                        help='reg for item-item graph')
    parser.add_argument('--device', nargs='?', default='cpu',
                        help='device')
    # cuda:0
    #可选参数：运行时输入：python demo.py --family=张 --name=三
    #   nargs：ArgumentParser对象通常将一个动作与一个命令行参数关联。nargs关键字参数将一个动作与不同数目的命令行参数关联在一起：
    #   nargs=N，一个选项后可以跟多个参数（action=’append’时，依然是一个选项后跟一个参数，只不过选项可以多次出现），参数的个数必须为N的值，这些参数会生成一个列表，当nargs=1时，会生成一个长度为1的列表。
    #   nargs=?，如果没有在命令行中出现对应的项，则给对应的项赋值为default。特殊的是，对于可选项，如果命令行中出现了此可选项，但是之后没有跟随赋值参数，则此时给此可选项并不是赋值default的值，而是赋值const的值。
    #   nargs=*，和N类似，但是没有规定列表长度。
    #   nargs=+，和*类似，但是给对应的项当没有传入参数时，会报错error: too few arguments。
    parser.add_argument('--num_epoch', type=int, default=500,
                        help='epoch number')
    parser.add_argument('--epoch', type=int, default=5,
                        help='frequency to evaluate')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='lambda')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='lr2')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='sigma')
    parser.add_argument('--f_epoch', type=int, default=20,
                        help='frontmodel epochs')
    # 向前预测遍历20次数据集
    parser.add_argument('--b_epoch', type=int, default=40,
                        help='backmodel epochs')
    # 反向更新参数遍历40次数据集
    parser.add_argument('--ite', type=int, default=5,
                        help='iterator')
    parser.add_argument('--f_max', type=int, default=10,
                        help='frontmodel iterator')
    parser.add_argument('--reuse', type=int, default=0,
                        help='if reuse past_domains')
    parser.add_argument('--pretrained', type=int, default=0,
                        help='if pretrained')
    # 是否有预训练？0为没有，其他为有预训练
    parser.add_argument('--wdi', type=int, default=2,
                        help='weight decay bias for item embedding')
    parser.add_argument('--sift', type=int, default=0,
                        help='if sift pos items')
    return parser.parse_args()


def init_attention_data(feat):
    fe = feat

    mask = np.load('/Users/taotao/Desktop/本地代码/mask.npy', allow_pickle=True)
    mask = torch.from_numpy(mask)

    # print("hi\tthe size of fe[0] = {}".format(fe.size(0)))
    # time.sleep(5)

    variant_feature = []
    print("Now start generting variant feature Matrix...\n")
    for i in tqdm(range(fe.size(0))):
        variant_feature.append(np.multiply(fe[i], mask).tolist())
    variant_feature = torch.FloatTensor(variant_feature)

    # 写文件的block:
    # fd = open('variant_feature.txt','w')
    # print("Now start wrire to file...\n")
    # for i in tqdm(range(fe.size(0))):
    #     s = variant_feature[i].tolist()
    #     for j in range(len(s)):
    #         if type(s[j]) != str:
    #             s[j] = str(s[j])
    #
    #     strs = ' '.join(s)
    #     fd.write(strs)
    # print("DONE~\n")
    # fd.close()

    print("Now total variant feature = \n {0} \n The size = ({1},{2})".format(variant_feature,
                                                                              len(variant_feature[0]),
                                                                              len(variant_feature[1])))
    return variant_feature

def re_write_mask():
    get_size = np.load()

    masks = np.array([0.5523, 0.5259, 0.5222, 0.5572, 0.5327, 0.5070, 0.5500, 0.5492, 0.5193,
            0.5253, 0.5332, 0.4896, 0.5381, 0.5486, 0.5384, 0.5155, 0.5169, 0.5520,
            0.5182, 0.5483, 0.5210, 0.5311, 0.5536, 0.5246, 0.5053, 0.5231, 0.5224,
            0.5452, 0.5878, 0.5132, 0.5371, 0.5449, 0.5574, 0.5425, 0.4957, 0.4715,
            0.5335, 0.5379, 0.5006, 0.5452, 0.4860, 0.5240, 0.5384, 0.5114, 0.5147,
            0.5179, 0.5337, 0.5109, 0.5339, 0.5208, 0.5157, 0.5156, 0.5030, 0.5171,
            0.5412, 0.5667, 0.5418, 0.5779, 0.5301, 0.5018, 0.5237, 0.5331, 0.5018,
            0.5221, 0.5412, 0.5186, 0.5697, 0.5200, 0.5374, 0.4886, 0.5462, 0.5385,
            0.5334, 0.5632, 0.5535, 0.5001, 0.5610, 0.5176, 0.5444, 0.5573, 0.5491,
            0.5457, 0.5399, 0.5795, 0.4930, 0.5522, 0.5268, 0.5256, 0.5224, 0.4933,
            0.5170, 0.5647, 0.5282, 0.5471, 0.5299, 0.5576, 0.5260, 0.5116, 0.5053,
            0.5105, 0.5310, 0.5316, 0.5313, 0.5211, 0.5433, 0.4741, 0.5302, 0.4994,
            0.5328, 0.5490, 0.5356, 0.5382, 0.5416, 0.5427, 0.4899, 0.4788, 0.5292,
            0.5111, 0.5119, 0.5341, 0.5110, 0.5220, 0.5238, 0.5114, 0.5268, 0.5035,
            0.5178, 0.5598, 0.6188, 0.6112, 0.6297, 0.5914, 0.5822, 0.6098, 0.5972,
            0.5793, 0.5546, 0.5439, 0.5836, 0.5087, 0.6384, 0.5124, 0.6331, 0.6191,
            0.6379, 0.6253, 0.5755, 0.6454, 0.6367, 0.6163, 0.5238, 0.6290, 0.6323,
            0.4405, 0.5312, 0.5224, 0.5356, 0.4422, 0.5724, 0.4324, 0.6181, 0.4321,
            0.6054, 0.5531, 0.5971, 0.5355, 0.4743, 0.5560, 0.5408, 0.6033, 0.6434,
            0.5992, 0.4383, 0.6331, 0.6397, 0.5495, 0.5566, 0.6159, 0.5573, 0.5036,
            0.6179, 0.5539, 0.5696, 0.4443, 0.6122, 0.6447, 0.4550, 0.6233, 0.6560,
            0.4388, 0.4753, 0.5810, 0.6089, 0.4624, 0.5718, 0.6314, 0.6283, 0.5277,
            0.6181, 0.6399, 0.6389, 0.6474, 0.5974, 0.5941, 0.6270, 0.6447, 0.5028,
            0.4349, 0.5705, 0.5839, 0.6426, 0.6031, 0.5620, 0.5096, 0.6088, 0.6536,
            0.6060, 0.4722, 0.6198, 0.4432, 0.6411, 0.5622, 0.6283, 0.6331, 0.5882,
            0.6428, 0.6067, 0.6445, 0.5770, 0.6447, 0.6117, 0.5690, 0.5810, 0.5622,
            0.6089, 0.6416, 0.5664, 0.5543, 0.5334, 0.6065, 0.6115, 0.6256, 0.6344,
            0.6195, 0.5579, 0.6275, 0.4503, 0.6084, 0.5772, 0.5515, 0.5226, 0.6330,
            0.6331, 0.6493, 0.4748, 0.6175, 0.6572, 0.5928, 0.6355, 0.6349, 0.6353,
            0.6695, 0.6037, 0.6250, 0.5876, 0.6446, 0.6170, 0.5594, 0.6146, 0.6268,
            0.5421, 0.6314, 0.5941, 0.5577, 0.6264, 0.5975, 0.6488, 0.6527, 0.5910,
            0.6111, 0.6245, 0.6489, 0.6085, 0.6041, 0.6553, 0.6085, 0.6204, 0.6403,
            0.5950, 0.6281, 0.6303, 0.6395, 0.6359, 0.6108, 0.5381, 0.6115, 0.6221,
            0.5269, 0.6162, 0.6224, 0.5490, 0.6288, 0.6160, 0.5294, 0.6583, 0.5966,
            0.6049, 0.5663, 0.6351, 0.6151, 0.5785, 0.6551, 0.6571, 0.6495, 0.6082,
            0.6212, 0.6220, 0.6025, 0.6219, 0.5652, 0.6332, 0.5718, 0.6434, 0.5699,
            0.6529, 0.5734, 0.6102, 0.6301, 0.5981, 0.6002, 0.6296, 0.5708, 0.6193,
            0.6039, 0.6597, 0.6034, 0.5672, 0.6290, 0.6162, 0.5913, 0.6254, 0.6155,
            0.6071, 0.6643, 0.6338, 0.6073, 0.5505, 0.5813, 0.5580, 0.6123, 0.6094,
            0.5500, 0.6313, 0.5396, 0.6448, 0.5937, 0.5418, 0.6394, 0.5797, 0.6368,
            0.6088, 0.5735, 0.6118, 0.6520, 0.6552, 0.6271, 0.6436, 0.5578, 0.5755,
            0.6347, 0.6170, 0.6345, 0.6267, 0.6242, 0.6599, 0.6558, 0.6137, 0.6050,
            0.6174, 0.5227, 0.6234, 0.6302, 0.6345, 0.6083])
    print(masks)
    np.save('/Users/taotao/Desktop/本地代码',masks)
    print("save .npy done")

args = parse_args()

args.p_emb = eval(args.p_emb)
args.p_embp = eval(args.p_embp)
args.p_ctx = eval(args.p_ctx)
args.p_proj = eval(args.p_proj)
args.p_w = eval(args.p_w)

if args.tolog == 0: #   输出所有output到stdio（输出到屏幕）
    logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
else:   #   输出output至log文件
    logfilename = 'logs/%s_%s.log' % (args.dataset, args.model)
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
    logging.info('log info to ' + logfilename)

logging.info(args)
if args.dataset == 'tiktok':
    ds = ds_tiktok(logging, args)
    #print("***\tThe shape of dataset = ",ds)
    # feature size = [76085, 384]
    # 76085 条, every feature have 384 特征
    fe = ds.get_data()
else:
    raise Exception('no dataset' + args.dataset)

variant_feat = init_attention_data(fe)

if args.model == 'UltraGCN':
    model = UltraGCN(ds, args, logging)
elif args.model == 'InvRL':
    model = InvRL(ds, args, logging)
elif args.model == 'MultAttention':
    print("Now Modifing The Config of model:\n")
    config = {
        "num_of_attention_heads": 2,# 这个属性是你想要划分出的几个层次
        "hidden_size": 384 # 隐藏特征数
    }
    # print('\t Num_of_attention_heads: 2\n\thidden_size: 4\n')
    MultAtt = MA.BertSelfAttention(config)
    # print(MultAtt)
    embed_rand2 = torch.rand((1, 3, 4))  # input
    print("the shape of orignal = \n {}".format(embed_rand2.shape))
    embed_rand = variant_feat
    print(embed_rand)
    print(f"Embed Shape: {embed_rand.shape}")
    print(f"Embed Values:\n{embed_rand}")
    output = MultAtt(embed_rand2)
    print(f"Output Shape: {output.shape}")
    print(f"Output Values:\n{output}")
else:
    raise Exception('unknown model type', args.model)

if args.model == 'InvRL':
    print("---=== Start process ERM learning ===---\n")
    model.train_erm()
model.train()
