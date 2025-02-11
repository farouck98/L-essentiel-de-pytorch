import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)
Y = torch.tensor([[3], [6], [9], [12], [15], [18], [21], [24], [27], [30]], dtype=torch.float32)

n, nb_features = X.shape

X_test = torch.tensor([100], dtype=torch.float32)

train_data = n
y = nb_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) :
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    

input_dim, output_dim = nb_features, nb_features

model = LinearRegression(input_dim, output_dim)

#Initialisation des paramètres
learning_rate = 0.01
n_epochs = 1000
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Boucles d'entraînement
for epoch in range(n_epochs):
    y_predicted = model(X)  # Prédiction du modèle
    l = loss(Y, y_predicted)  # Calcul de la perte
    
    l.backward()  # Calcul des gradients

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        w, b = model.parameters()
        print(f"At epoch {epoch+1}: w = {w[0][0].item():.3f}, b = {b.item():.3f}, loss = {l.item():.3f}")
print(f"Après les opérations d'entraînement: F({X_test.item()}) = {model(X_test).item()}")
