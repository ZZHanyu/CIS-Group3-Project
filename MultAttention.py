import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

# torch.set_printoptions(profile="full")

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = ""

        if(torch.cuda.is_available() == True):
            print("***\tAttention model using CUDA...\n")
            self.device = "cuda:0"
        else:
            print("***\tCUDA not found...Now using CPU\n")
            self.device = "cpu"


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
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        # print("1. Mixed_value_layer = {}\n".format(mixed_value_layer.shape))
        # print("2. Mixed_key_layer = {}\n".format(mixed_key_layer.shape))
        # print("3. Mixed_value_layer = {}\n".format(mixed_query_layer))

        query_layer = self.transpose_for_scores(mixed_query_layer).to(self.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        # [Batch_size= 1, Num_of_heads = 2, Seq_length = 76085, Head_size= 192]
        # print(query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer).to(self.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer).to(self.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2)).to(self.device)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size).to(self.device)   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores).to(self.device)   # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,value_layer).to(self.device)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return output


