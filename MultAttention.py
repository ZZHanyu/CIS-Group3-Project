import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from tiktok.load_data import dataset as ds_tiktok

# torch.set_printoptions(profile="full")

class BertSelfAttention(nn.Module):
    def __init__(self, config, ds, args, logging):
        super().__init__()
        self.feat = ds
        self.args = args
        self.logging = logging

        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"
        # hidden size = 384
        # num_of_attention_heads = 2

        self.num_attention_heads = config['num_of_attention_heads'] # = 2
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads']) # = 384 / 2 = 192
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # nn.Linear(in.feat:输入的向量大小, out.feat:输出的向量大小)
        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

        # 数据初始化Part 2:
        variant_feat = self.init_attention_data(self.feat)
        data_list = variant_feat.chunk(21, 0)
        variant_feat = torch.stack(data_list, 0)
        embed_rand = variant_feat
        embed_rand = embed_rand.to(self.args.device)
        print(f"Embed Shape: {embed_rand.shape}")
        print(f"Embed Values:\n{embed_rand}")

        self.hidden_states = embed_rand


    def transpose_for_scores(self, x):
        # print("hi\t the x is = \n {}".format(x))
        # print("The shape of x = \n {}".format(x.shape))
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        y = x.permute(0, 2, 1, 3) # 把x维度的顺序更改，0，2，1，3分别为原始顺序index

        # print("\thi the y is = \n {}".format(y))
        # print("\tThe shape of y = \n {}".format(y.shape))
        return  y

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states).to(self.args.device)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states).to(self.args.device)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states).to(self.args.device)  # [Batch_size x Seq_length x Hidden_size]
        # print("1. Mixed_value_layer = {}\n".format(mixed_value_layer.shape))
        # print("2. Mixed_key_layer = {}\n".format(mixed_key_layer.shape))
        # print("3. Mixed_value_layer = {}\n".format(mixed_query_layer))

        query_layer = self.transpose_for_scores(mixed_query_layer).to(self.args.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        # [Batch_size= 1, Num_of_heads = 2, Seq_length = 76085, Head_size= 192]
        # print(query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer).to(self.args.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer).to(self.args.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2)).to(self.args.device)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        attention_scores = (attention_scores / math.sqrt(self.attention_head_size)).to(self.args.device)   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        attention_probs = nn.Softmax(dim=-1)(attention_scores).to(self.args.device)   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        context_layer = torch.matmul(attention_probs,value_layer).to(self.args.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)
        # reshape the output value:
        # after 合并: [76083, 384]
        transfer_output = torch.flatten(output, start_dim=0, end_dim=1)

        #打印输出
        print("***Shape of mutiattention = {} \n Shape of ds = {}".format(output.shape, fe.shape))
        # Shape of fe = torch.Size([76085, 384])
        # Output Shape: torch.Size([21, 3623, 384])
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n{output}")

        # torch.set_printoptions(profile="full")
        print(f"After transfomation = {transfer_output}")
        print(f"Feature = {self.feat}")
        print("***\tAttention Model Sucessfully\t***\n")

        return transfer_output

    def init_data(feat):
        fe = feat
        mask = np.load('/Users/taotao/Desktop/本地代码/mask.npy', allow_pickle=True)
        mask = torch.from_numpy(mask)

        # print("hi\tthe size of fe[0] = {}".format(fe.size(0)))
        # time.sleep(5)

        variant_feature = []
        print("***\tNow start generting variant feature Matrix...\n")
        for i in tqdm(range(fe.size(0))):
            variant_feature.append(np.multiply(fe[i], mask).tolist())
        variant_feature = torch.FloatTensor(variant_feature)

        # 文件IO
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

        # print("Now total variant feature = \n {0} \n The size = ({1},{2})".format(variant_feature,
        #                                                                           len(variant_feature[0]),
        #
        #                                                                           len(variant_feature[1])))
        return variant_feature


