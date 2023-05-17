#yangqiok
import argparse
import sys
import logging
from tiktok.load_data import dataset as ds_tiktok
import numpy as np
import torch

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
    parser.add_argument('--bsz', type=int, default=256,
                        help='batch size')
    # 批量大小
    parser.add_argument('--ssz', type=int, default=256,
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
    parser.add_argument('--device', nargs='?', default='cuda:0',
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
    parser.add_argument('--path', nargs='?',default='C:\\Users\\vipuser\\Desktop\\509run\\mask.npy',
                        help='the mask file path')
    return parser.parse_args()

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
    fe = ds.get_data()
    # feature is a torch
    # print("***\tThe shape of dataset = ",ds)
    # feature size = [76085, 384]
    # 76085 条, every feature have 384 特征

else:
    raise Exception('no dataset' + args.dataset)

if args.model == 'UltraGCN':
    model = UltraGCN(ds, args, logging)
    model.train()
elif args.model == 'IRAT':
    model = InvRL(ds, args, logging)
    model.train()
else:
    raise Exception('unknown model type', args.model)
