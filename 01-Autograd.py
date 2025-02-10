import torch

X = torch.rand(4, requires_grad=True)
print(f"X={X}")
print(f"Grad_fn = {X.grad_fn}")



Y = X * 2
print(f"Y={Y}")
print(f"Grad_fn = {Y.grad_fn}")
"""
Y = X ** 2
print(f"Y={Y}")
print(f"Grad_fn = {Y.grad_fn}")
"""


Z = Y.mean()
print(f"Z={Z}")
print(f"Grad_fn = {Z.grad_fn}")

Y.retain_grad()

print("Grad avant la propagation :")
print(f"Y={Y}")
print(f"Grad de Y = {Y.grad}")
print(f"X={X}")
print(f"Grad de X = {X.grad}")

#L'appel de cette méthode va propager le calcul du gradient de Z vers X

Z.backward()

print("Grad après la propagation :")
print(f"Y={Y}")
print(f"Grad de Y = {Y.grad}")
print(f"X={X}")
print(f"Grad de X = {X.grad}")
