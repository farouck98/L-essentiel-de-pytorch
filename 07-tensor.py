import torch

a1 = torch.ones(3)
a2 = torch.zeros(3)

b = a1 + a2
c = 2 * a1

print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"b = {b}")
print(f"c = {c}")

#Modification du tensor c
c.add_(a1)
print(f"c = {c}")