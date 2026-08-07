import torch
# import matplotlib as plt

with open("names.txt", encoding="utf-8") as f:
    name_main = [line.strip() for line in f if line.strip()]

chars = [chr(i) for i in range(ord('a'), ord('z')+1)] + ['.']

Axis_r = {s: i for i, s in enumerate(chars)}

def process_table(nameDB: list[str]):

    Array = torch.zeros((27,27), dtype = torch.int32)

    log_to_assess_quality = 0.0
    n = 0

    for sample in nameDB:

        sample = "." + sample + "."

        for letters in range(len(sample)-1):

            axisX = Axis_r[sample[letters]]
            axisY = Axis_r[sample[letters + 1]]

            Array[axisX,axisY] += 1

            prob = Array[axisX,axisY]

            logprob = torch.log(prob)

            NLL = -logprob

            log_to_assess_quality += NLL

            n += 1

    Array = Array + 1
    print(f'{log_to_assess_quality/n=}')

    return Array

def loss_function(Refrence_table: torch.Tensor, NLL = int):
    torch.log(Refrence_table)

def output(Refrence_table: torch.Tensor, min_length: int = 3):

    Array = Refrence_table.float()
    Array /= Array.sum(1, keepdim=True)

    output = []

    current_idx = Axis_r["."]

    # while True:

    #     weights = Array[current_idx].clone()

    #     if weights.sum() == 0:
    #         break  

    #     if len(output) < min_length:
    #         weights[Axis_r["."]] = 0.0

    #     next_idx = torch.multinomial(weights, num_samples=1, replacement= True).item()

    #     if next_idx == Axis_r["."]:
    #         break

    #     output.append(chars[next_idx])

    #     current_idx = next_idx

    print(Array)


process_table(name_main)