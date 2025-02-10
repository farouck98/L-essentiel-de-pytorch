import torch

L = [1,2,3,4,5]
a1 = torch.tensor(L, dtype = torch.float32, requires_grad=True)

print(f"a1 = {a1}")
