import torch
import numpy as np

"""A = torch.rand(5)
print(f"A = {A}")

B = A.numpy()
print(f"B = {B}")
print(f"type(B) = {type(B)}")

#la modification d'une valeur du tableau Numpy B impacte aussi sur le tenson A
B[0] = 99
print(f"A = {A}")"""


A = np.ones(5)

#Création d'un tensor à partir d'un tableau numpy A
B = torch.from_numpy(A)
C = torch.tensor(A)

print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")

#Essayons de faire des calculs
A*=10

print("===========================")
print(f"A = {A}")
print(f"B = {B}")

#Nous constatons que le calcul n'a par été effectuer sur le tensor C qui a été créé
#avec tensor à partir du tableau numpy A
print(f"C = {C}")