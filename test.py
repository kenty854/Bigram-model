import torch

itos = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.']
stoi = {s: i for i, s in enumerate(itos)}

context_length = 3
emb_size = 10

params = torch.load("params.pt")
emb_table = params['embedding']
W1 = params['W1']
b1 = params['b1']
W2 = params['W2']
b2 = params['b2']

def generate_name(max_len=20):

    context = [0] * context_length
    out = []

    for _ in range(max_len):

        emb = emb_table[torch.tensor([context])]
        x = emb.view(1, emb_size * context_length)

        h = torch.tanh(x @ W1 + b1)
        logits = h @ W2 + b2

        probs = torch.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1).item()

        if idx == stoi['.']:
            break

        out.append(itos[idx])
        context = context[1:] + [idx]

    return ''.join(out)

if __name__ == "__main__":
    for _ in range(10):
        print(generate_name())