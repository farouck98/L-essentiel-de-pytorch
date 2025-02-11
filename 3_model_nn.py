import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 3
batch_size = 100

norm_transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

x_train = torchvision.datasets.CIFAR10(root="./cifar10/data",
                                       train=True,
                                       transform=norm_transform,
                                       download=True)

x_test = torchvision.datasets.CIFAR10(root="./cifar10/data",
                                      train=False,
                                      download=True,
                                      transform=norm_transform)

train_data_loader = torch.utils.data.DataLoader(dataset=x_train,
                                                batch_size=batch_size,
                                                shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=x_test,
                                               batch_size=batch_size,
                                               shuffle=False)

cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(3, 32, 3)
        self.pool_layer = nn.MaxPool2d(2, 2)
        self.conv_layer_2 = nn.Conv2d(32, 64, 3)
        self.conv_layer_3 = nn.Conv2d(64, 128, 3)
        self.lin_layer_1 = nn.Linear(128*4*4, 128)
        self.lin_layer_2 = nn.Linear(128, 64)
        self.lin_layer_3 = nn.Linear(64, 10)

    def forward(self, input_data):
        l = self.conv_layer_1(input_data)
        l = F.relu(l)
        l = self.pool_layer(l)
        l = self.conv_layer_2(l)
        l = F.relu(l)
        l = self.pool_layer(l)
        l = self.conv_layer_3(l)
        l = F.relu(l)
        l = torch.flatten(l, 1)
        l = self.lin_layer_1(l)
        l = F.relu(l)
        l = self.lin_layer_2(l)
        l = F.relu(l)
        l = self.lin_layer_3(l)

        return l

conv_nn = ConvNN().to(device)

loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(conv_nn.parameters(), lr = 0.01)

nb_images = len(train_data_loader)

for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = conv_nn(images)
        loss = loss_criterion(outputs, labels)


        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        optimizer.zero_grad()

    print(f"Epochs {epoch+1}/{epochs}, loss = {epoch_loss / nb_images:.3f}")

print("Fin de la phase d'entraînement !")

PATH = "./cnn.pth"
torch.save(conv_nn.state_dict(), PATH)

########################################################################

loaded_conv_nn = ConvNN()
loaded_conv_nn.load_state_dict(torch.load(PATH))
loaded_conv_nn.to(device)
loaded_conv_nn.eval()

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    test_data_size = len(test_data_loader.dataset)

    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = conv_nn(images)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_conv_nn(images)

    acc = 100.0 * n_correct / test_data_size
    print(f"Accuracy = {acc}")

    acc = 100.0 * n_correct2 / test_data_size
    print(f"Accuracy = {acc}")
