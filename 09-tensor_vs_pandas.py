import torch
import pandas as pd

#lecture et affcichage du fichier csv
df = pd.read_csv('tensor_vs_panda.csv')
print(df)

#Créer un tensor à partir d'un dataframe pandas
tensor = torch.tensor(df.values)
print("Tensor : ")
print(tensor)
print(f"Shape : {tensor.shape}")
