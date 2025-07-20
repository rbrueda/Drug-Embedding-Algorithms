# Source code: https://github.com/ZihuiCheng/MI-DDI/blob/main/newmodels.py

import torch
#from torch import nn
import pandas as pd
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)

from torch_geometric.nn.inits import reset, zeros
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
import math
from torch.nn import Linear, GRU, Parameter
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set, NNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_
from torch import Tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
         tensor.data.fill_(0)

class BaseGGNN(MessagePassing):
    def __init__(self, state_size: int, num_layers: int,
                 aggr: str = 'add', bias: bool = True,
                 total_edge_types: int = 10,
                 use_resnet=False, subnum: int = 20, atomnum: int = 20):
        super(BaseGGNN, self).__init__(aggr=aggr)

        self.state_size = state_size
        self.out_channels = state_size
        self.num_layers = num_layers
        self.use_resnet = use_resnet
        self.subnum = subnum
        self.atomnum = atomnum

        self.weight = nn.Parameter(Tensor(num_layers, state_size, state_size))

        # Edge-type weights
        self.edge_type_weight = nn.Parameter(torch.zeros(total_edge_types, state_size, state_size))
        self.edge_type_bias = nn.Parameter(torch.zeros(1, state_size))

        # Update mechanism
        if self.use_resnet:
            self.mlp_1 = nn.Sequential(
                nn.Linear(state_size, state_size),
                nn.ReLU(inplace=True),
                nn.Linear(state_size, state_size),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(state_size, state_size),
                nn.ReLU(inplace=True),
                nn.Linear(state_size, state_size),
            )
        else:
            self.rnn = nn.GRUCell(state_size, state_size, bias=bias)

        self.reset_parameters()

        # CSV SMILES dictionary
        df = pd.read_csv('data/input/cleaned_drugbank_smiles_mapping.csv', header=0)
        self.dic = dict(zip(df["DrugBank_ID"], df["SMILES"]))

        # Final prediction MLP (if used)
        self.mlp = nn.ModuleList([
            nn.Linear(2 * (self.subnum + self.atomnum), 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 65)
        ])

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_type_weight)
        glorot(self.edge_type_bias)

        if self.use_resnet:
            for layer in self.mlp_1:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.mlp_2:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.rnn.reset_parameters()

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return x_j
        weighted = torch.einsum("ab,bcd->acd", (edge_attr, self.edge_type_weight))
        msg = torch.bmm(weighted, x_j.unsqueeze(-1)).squeeze(-1)
        return msg + self.edge_type_bias.repeat(x_j.size(0), 1)

    def get_weight(self, triples, batch_size):
        h_data, _, _ = triples
        x1, edge_index1, edge_attr1, batch1 = h_data.x, h_data.edge_index, h_data.edge_attr, h_data.batch

        if x1.size(-1) > self.out_channels:
            raise ValueError("Input feature dimension > output channels.")

        if x1.size(-1) < self.out_channels:
            pad = x1.new_zeros(x1.size(0), self.out_channels - x1.size(-1))
            x1 = torch.cat([x1, pad], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x1, self.weight[i])
            m = self.propagate(edge_index1, x=m, edge_attr=edge_attr1)
            if self.use_resnet:
                x1 = self.mlp_2(m + self.mlp_1(x1))
            else:
                x1 = self.rnn(m, x1)

        x1 = global_mean_pool(x1, batch1)
        repr_h = x1  # already pooled to (num_molecules, state_size)
        # Convert tensor to numpy array → DataFrame
        emb_np = repr_h.cpu().detach().numpy()
        df = pd.DataFrame(emb_np)

        # Save as Parquet
        #df.to_parquet(f'MPNN-Embeddings.pq', index=False)
        return repr_h

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def __repr__(self):
        return f'{self.__class__.__name__}({self.out_channels}, num_layers={self.num_layers})'


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        # if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states


