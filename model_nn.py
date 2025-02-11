import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#Détection du matériel GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Paramètres de l'entraînement

nb_classes = 10
epochs = 2
batch_size = 100

#Chargement des données d'entraînement
x_train = torchvision.datasets.MNIST(root="./data",
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

#Chargement des données de test
x_test = torchvision.datasets.MNIST(root="./data",
                                    train=False,
                                    transform=transforms.ToTensor())

#Préparation du Dataloader
train_data_loader = torch.utils.data.DataLoader(dataset=x_train,
                                                batch_size=batch_size,
                                                shuffle=True)

#Création du DataLoader pour les données de test
x_test_loader = torch.utils.data.DataLoader(dataset=x_test,
                                            batch_size=batch_size,
                                            shuffle=False)
"""
#Récupération d'un batch d'exemples
examples = iter(x_test_loader)

#Extraction du premier batch
images, labels = next(examples)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0])
plt.show()
"""

class SimpleNN(nn.Module):
    def __init__(self, nb_classes):
        super(SimpleNN, self).__init__()
        self.lin_layer_1 = nn.Linear(784, 500)
        self.relu = nn.ReLU()
        self.lin_layer_2 = nn.Linear(500, nb_classes)

    def forward(self, input_data):
        l1 = self.lin_layer_1(input_data)
        l1 = self.relu(l1)
        l2 = self.lin_layer_2(l1)
        return l2

simple_nn = SimpleNN(nb_classes).to(device)

loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(simple_nn.parameters(), lr=0.01)

nb_images = len(train_data_loader)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
    
        outputs = simple_nn(images)
        loss = loss_criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % batch_size == 0:
            print(f"Epochs {epoch+1}/{epochs}, sample : {i+1}/{nb_images} : loss ={loss.item():.3f} , ")


with torch.no_grad():
    n_correct = 0
    test_data_size = len(x_test_loader.dataset)

    for images, labels in x_test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = simple_nn(images)

        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / test_data_size
    print(f"Accuracy = {100*acc}")
