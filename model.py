import torch
import torch.nn as nn
from tokenizer import Tokenizer
from data.bookCorpus import Bookcorpus
from torch.utils.data import DataLoader

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



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

        blocks = []
        for _ in range(LAYERS):
            blocks.append(Block())
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, input):
        batch_size, seq_length = input.size()

        input = self.emb(input)

        pos_ids = torch.arange(seq_length)
        pos_embs = self.pos_emb(pos_ids)

        input = torch.add(input, pos_embs)
        for block in self.blocks:
            input = block(input)
        
        input = self.ln(input)
        return self.lm_head(input)


m = Model()
m.train()

t = Tokenizer()

builder = Bookcorpus()

builder.download_and_prepare()

ds = builder.as_dataset(split='train[:10%]')

train_dataloader = DataLoader(ds, batch_size=32, shuffle=True)

EPOCHS = 3
LR = 1e-3

opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for i in range(EPOCHS):
    print(f"Epoch: {i+1}")
    for i, batch in enumerate(train_dataloader):
        token_list = []
        max_len = 0
        for sentance in batch["text"]:
            cur_token_list = t.tokenize(sentance)
            if len(cur_token_list) > max_len:
                max_len = len(cur_token_list)
            token_list.append(cur_token_list)

        for sentance_tokens in token_list:
            while len(sentance_tokens) < max_len:
                sentance_tokens.append(t.end_of_file_index)
        token_list = torch.tensor(token_list)
        logits = m(token_list)
        print(max_len)
        print(logits[:, :max_len-1, :].size())
        print(token_list[:, 1:].size())
        loss = loss_fn(logits[:, :max_len-1, :].transpose(-1, -2), token_list[:, 1:])
        print(loss.item())
        loss.backward()
        opt.step()

torch.save(m.state_dict(), "model.pt")


