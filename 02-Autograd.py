import torch

T =torch.rand(5, requires_grad=True)

X = T**2
Y=X**2
Z = Y.mean()
Y.retain_grad()
X.retain_grad()

print("======Grad avant la propagation :======")
print(f"Y={Y}")
print(f"Grad de Y= {Y.grad}")
print(f"X={X}")
print(f"Grad de X = {X.grad}")
print(f"T={T}")
print(f"Grad de T = {T.grad}")

Z.backward()

print("======Grad après la propagation :===========")
print(f"Y={Y}")
print(f"Grad de Y = {Y.grad}")
print(f"X={X}")
print(f"Grad de X = {X.grad}")
print(f"T={T}")
print(f"Grad de T = {T.grad}")