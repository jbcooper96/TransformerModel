import torch
import torch.nn as nn
from tokenizer import Tokenizer
from data.bookCorpus import Bookcorpus
from torch.utils.data import DataLoader
import math

MAX_SEQUENCE_LENGTH = 500
VOCAB_SIZE = 5000
EMBEDDING_DIM = 1000
NUM_HEADS = 10
LAYERS = 6

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBEDDING_DIM, NUM_HEADS, batch_first=True)
        self.c_attn = nn.Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM)
        self.ln_1 = nn.LayerNorm(EMBEDDING_DIM)
        self.ln_2 = nn.LayerNorm(EMBEDDING_DIM)

        self.lin1 = nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM)
        self.lin2 = nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(.5)

    def forward(self, input):
        residual = input
        input = self.ln_1(input)
        query, key, value = self.c_attn(input).split(EMBEDDING_DIM, dim=2)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(input.size()[1])
        input = self.attn(query, key, value, is_causal=True, need_weights=False, attn_mask=causal_mask)[0]
        input = input + residual

        residual = input
        input = self.ln_2(input)
        input = self.lin1(input)
        input = self.activation(input)
        input = self.lin2(input)
        input = self.dropout(input)
        return input + residual


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    
        pe = torch.zeros(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        position = torch.arange(0, MAX_SEQUENCE_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, EMBEDDING_DIM, 2).float() * -(math.log(10000.0) / EMBEDDING_DIM))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
    
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.pos_emb = PositionalEmbedding()

        blocks = []
        for _ in range(LAYERS):
            blocks.append(Block())
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, input):
        batch_size, seq_length = input.size()

        input = self.emb(input)
        input = self.pos_emb(input)

        for block in self.blocks:
            input = block(input)
        
        input = self.ln(input)
        return self.lm_head(input)


