import torch
from model_1 import chars, Axis_r

weights_local = torch.load("model.pt")

def gen_name():
    name = "."
    while True:
        logits = torch.nn.functional.one_hot(torch.tensor(Axis_r[name[-1]]), num_classes=27).float() @ weights_local
        prob = logits.softmax(dim=0)
        next_char_idx = torch.multinomial(prob, num_samples=1)
        next_char = chars[next_char_idx]
        name += next_char
        if next_char == ".":
            break
    return name[1:-1]

for _ in range(100):
    print(gen_name())