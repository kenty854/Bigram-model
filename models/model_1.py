import torch

with open("names.txt", encoding="utf-8") as f:
    name_main = [line.strip() for line in f if line.strip()]

chars = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['.']
Axis_r = {s: i for i, s in enumerate(chars)}

def create_dataset(names: list[list]):
    ctx = []
    ans = []

    for name in names:
        name = "." + name + "."
        for i in range(len(name)-1):
            ctx.append(Axis_r[name[i]])
            ans.append(Axis_r[name[i+1]])

    return ctx, ans


def training(names):
    torch.manual_seed(42)

    try: 
        weights = torch.load("model.pt")\
        
        print("Loaded existing model weights.")

    except FileNotFoundError:

        print("No existing model weights found. Initializing new weights.")
        
        weights = torch.randn((27, 27), requires_grad=True)

    ctx, ans = create_dataset(names)

    for epoch in range(10):

        total_loss = torch.zeros((), dtype=torch.float32)

        for b in range(len(ctx)):

            one_hot = torch.nn.functional.one_hot(

                torch.tensor(ctx[b], dtype=torch.long), num_classes=27

            ).float()

            logits = one_hot @ weights

            prob = logits.softmax(dim=0)

            loss = -torch.log(prob[ans[b]])

            total_loss = total_loss + loss

        average_loss = total_loss / len(ctx)
        average_loss.backward()

        with torch.no_grad():
            weights -= 0.1 * weights.grad

        weights.grad = None
        print(f"epoch {epoch + 1}: loss = {average_loss.item():.4f}")

    torch.save(weights, "model.pt")
    return weights


if __name__ == "__main__":
    training(name_main)