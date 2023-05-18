import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from tiktok.load_data import dataset as ds_tiktok

class Attention(nn.Module):
    def __init__(self, config, fe, args, logging):
        super().__init__()
        self.feat = fe
        self.args = args
        self.logging = logging

        # 数据初始化Part 2:
        data_list = self.feat.chunk(5, 0)
        variant_feat = torch.stack(data_list, 0)
        embed_rand = variant_feat
        # embed_rand = embed_rand
        print(f"Embed Shape: {embed_rand.shape}")
        print(f"Embed Values:\n{embed_rand}")

        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"
        # hidden size = 128
        # num_of_attention_heads = 2

        self.num_attention_heads = config['num_of_attention_heads'] # = 4
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads']) # = 128 / 4 = 31
        self.all_head_size = self.num_attention_heads * self.attention_head_size # = 128

        # nn.Linear(in.feat:输入的向量大小, out.feat:输出的向量大小)
        self.query = nn.Linear(config['hidden_size'], self.all_head_size) # 128, 128
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])


        self.hidden_states = embed_rand
        self.forward()


    def transpose_for_scores(self, x):
        # print("hi\t the x is = \n {}".format(x))
        # print("The shape of x = \n {}".format(x.shape))
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        y = x.permute(0, 2, 1, 3) # 把x维度的顺序更改，0，2，1，3分别为原始顺序index

        # print("\thi the y is = \n {}".format(y))
        # print("\tThe shape of y = \n {}".format(y.shape))
        return  y

    def forward(self):
        mixed_query_layer = self.query(self.hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(self.hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(self.hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        # print("1. Mixed_value_layer = {}\n".format(mixed_value_layer.shape))
        # print("2. Mixed_key_layer = {}\n".format(mixed_key_layer.shape))
        # print("3. Mixed_value_layer = {}\n".format(mixed_query_layer))

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        # [Batch_size= 1, Num_of_heads = 2, Seq_length = 76085, Head_size= 192]
        # print(query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        attention_scores = (attention_scores / math.sqrt(self.attention_head_size))   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        attention_probs = nn.Softmax(dim=-1)(attention_scores)   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # torch.cuda.empty_cache()
        context_layer = torch.matmul(attention_probs,value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)
        # reshape the output value:
        # after 合并: [76083, 384]
        transfer_output = torch.flatten(output, start_dim=0, end_dim=1)

        #打印输出
        print("***Shape of mutiattention = {} \n Shape of ds = {}".format(output.shape, self.feat.shape))
        # Shape of fe = torch.Size([76085, 384])
        # Output Shape: torch.Size([21, 3623, 384])
        print(f"Output Shape: {output.shape}")
        print(f"Output Values:\n{output}")

        # torch.set_printoptions(profile="full")
        print(f"After transfomation = {transfer_output}")
        print(f"Feature = {self.feat}")
        print("***\tAttention Model Sucessfully\t***\n")

        transfer_output = torch.mul(transfer_output, self.feat)

        return transfer_output



