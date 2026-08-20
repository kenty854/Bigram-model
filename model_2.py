import torch
import time

chars = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.']
Axis_r = {s: i for i, s in enumerate(chars)}

itos = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.']
stoi = {s: i for i, s in enumerate(itos)}

with open("names.txt", encoding="utf-8") as f:
    name_main = [line.strip() for line in f if line.strip()]

def create_dataset(names):

    contexts = []
    ans = []

    context_length = 3

    for name in names:
    
        context = [0] * context_length

        for letter in name + ".":

            current_idx = stoi[letter]

            contexts.append(context)

            ans.append(current_idx)

            context = context[1:] + [current_idx]

    return contexts, ans

def training(contexts, ans):

    X = torch.tensor(contexts)
    Y = torch.tensor(ans)

    ix = torch.randint(0, X.shape[0], (32,))
    Xb = X[ix]
    Yb = Y[ix]

    epoch = 50000
    lr = 0.01

    neuron_count = 200
    emb_size = 10
    try:
        params = torch.load("params.pt")
        emb_table = params['embedding'].requires_grad_()
        W1 = params['W1'].requires_grad_()
        b1 = params['b1'].requires_grad_()
        W2 = params['W2'].requires_grad_()
        b2 = params['b2'].requires_grad_()

        print("Loaded existing model parameters.")

    except FileNotFoundError:

        print("No existing model parameters found. Initializing new parameters.")

        emb_table = torch.randn((27, emb_size), requires_grad=True)

        emb_table = torch.randn((27, emb_size), requires_grad=True)

        W1 = (
            torch.randn((emb_size * 3, neuron_count))
            * (5/3)
            / ((emb_size * 3) ** 0.5)
        ).requires_grad_()

        b1 = torch.zeros(neuron_count, requires_grad=True)

        W2 = (torch.randn((neuron_count, 27)) * 0.01).requires_grad_()

        b2 = torch.zeros(27, requires_grad=True)

    for rnd in range(epoch):

        if rnd % 100 == 0:
            t0 = time.time()

        emb = emb_table[Xb]
        
        x = emb.view(-1, emb_size * 3)  

        h = torch.tanh(x @ W1 + b1)
        logits = h @ W2 + b2

        loss = torch.nn.functional.cross_entropy(logits, Yb)


        emb_table.grad = None
        W1.grad = None
        b1.grad = None
        W2.grad = None
        b2.grad = None


        loss.backward()


        with torch.no_grad():
            emb_table -= lr * emb_table.grad
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad

        if rnd % 100 == 99:
            print(
                f"{rnd-99}-{rnd} | "
                f"loss={loss.item():.4f} | "
                f"time={time.time()-t0:.4f}s"
            )

    save_yn = input("Do you want to save the model parameters? (y/n): ")

    if save_yn.lower() == 'y':
        torch.save({

            'embedding': emb_table.detach(),

            'W1': W1.detach(), 
            'b1': b1.detach(),

            'W2': W2.detach(), 
            'b2': b2.detach(),
    
        }, 'params.pt')



if __name__ == "__main__":
    contexts, ans = create_dataset(name_main)
    training(contexts, ans)