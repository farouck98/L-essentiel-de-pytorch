import torch

"""a1 = torch.ones(3)
a2 = torch.zeros(3)

b = a1 + a2
c = 2 * a1

print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"b = {b}")
print(f"c = {c}")

#Modification du tensor c
c.add_(a1)
print(f"c = {c}")"""

#génération d'un tensor d'une matrice de 5 lignes et 5 colonnes
a3 = torch.rand(5,5)
print(f"a3 = {a3}")

#accéder à la première colonne de la matrice
print(f"a3[:, 0] = {a3[:, 0]}")

#accéder à la valeur se trouvant à la première ligne et à la deuxième colonne de la matrice
print(f"a3[0:, 1] = {a3[0, 1]}")

