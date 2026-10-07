import torch
from pathlib import Path

text_path = Path(__file__).with_name("shakespear.txt")
with text_path.open("r", encoding="utf-8") as f:
    text = f.read()

#-PARAMETERS-#
Batch = 4
context_window = 10
emb_dim = 32
Iters = 3000
lr = 1e-2
#------------#


#TOKENIZATION
chars = sorted(list(set(text)))
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
input = torch.tensor(encode(text), dtype=torch.long)


# DATA SPLIT
n = int(0.8 * len(input))
train_data = input[:n]
val_data = input[n:]


class Transformer(torch.nn.Module):
    def __init__(self, input):
        self.vocab_size = len(chars)
        self.emb_dim = emb_dim
        self.cxt_win = context_window
        self.emb_mat = torch.randn((len(chars), emb_dim), dtype=torch.float32)
        self.pos_emb = torch.randn((self.cxt_win , self.emb_dim))
        self.W_Q = torch.randn((emb_dim, emb_dim), dtype=torch.float32)
        self.W_K = torch.randn((emb_dim, emb_dim), dtype=torch.float32) 
        self.W_V = torch.randn((emb_dim, emb_dim), dtype=torch.float32)
        super().__init__()

    def forward(self):
        for a in range(Iters):
