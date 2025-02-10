import torch
import numpy as np

#Deux échantillons 
X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
Y = torch.tensor([3, 6, 9, 12, 15, 18, 21, 24, 27, 30], dtype=torch.float32)


#Nous avons voulu estier les fonction qui lie X à Y à savoir Y = w * X

w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

def get_prediction(x):
    return (w * x)

def compute_loss(y_predicted, y_observed):
    total_loss = (y_predicted - y_observed) ** 2
    mean_loss = total_loss.mean()
    return mean_loss

learning_rate = 0.01
max_epochs = 100
for epoch in range(max_epochs):
    y_predicted = get_prediction(X)
    current_loss = compute_loss(y_predicted, Y)
    current_loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad

    print(f"Epoch: {epoch} | Loss = {current_loss:.3f} | W Grad={w.grad:.3f} | W = {w.item():.3f}")
    w.grad.zero_()
