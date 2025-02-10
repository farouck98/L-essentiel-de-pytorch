import torch

l1 = [1,2,3,4,5]
a1 = torch.tensor(l1)
print(f"a1 = {a1}")
print(f"a1 data type = {a1.dtype}")

l2 = [1,2,3,4, 5.2]
a2 = torch.tensor(l2, dtype=torch.float16)
print(f"a2 data type = {a2.dtype}")