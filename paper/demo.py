import torch
import torch.nn.functional as F
from torch import nn

if __name__ == "__main__":

    for i in range(1, 10):
        print()
        for j in range(1, i+1):
            print(f"{j} * {i} = {i*j}", end='\t')

