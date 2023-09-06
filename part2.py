# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR_Classifier(nn.Module):
    def __init__(self):
        super(CIFAR_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
         # -> n, 3, 32, 32
        x = self.pool(torch.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(torch.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = torch.relu(self.fc1(x))               # -> n, 120
        x = torch.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
    # architecture from https://www.youtube.com/watch?v=pDdP0TFzsoQ

"""
The batch size represents the number of data samples (or images) that are processed in a single forward and backward pass through the neural network during each training iteration.
"""
batch_size = 5

"""
An epoch is one complete pass through the entire training dataset. During one epoch, the model processes and learns from all the training examples once.
"""
num_epochs = 4
"""
In each iteration, the gradients of the loss function with respect to the parameters are computed. The learning rate is then used to scale these gradients before applying them to update the parameters.
parameter_new = parameter_old - (learning_rate * gradient)
"""
learning_rate = 0.001

cifar_10_file_dir = "./CIFAR-10"

# Define data transforms (you can customize these as needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to mean 0 and std 1
])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root=cifar_10_file_dir, train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root=cifar_10_file_dir, train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Initialize the model, loss function, and optimizer
model = CIFAR_Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)

loss_at_epoch = {}
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + batch_size) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

        running_loss += loss.item()
    loss_at_epoch[epoch] = average_loss = running_loss / len(train_loader)

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():

    correct = 0
    total = 0

    # Initialize a dictionary to store class-wise accuracy
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for batch in test_loader:
        test_X, test_y = batch
        test_X, test_y = test_X.to(device), test_y.to(device)

        # Forward pass
        test_outputs = model(test_X)
        _, predicted = torch.max(test_outputs, 1)

        # Compute overall accuracy
        correct += (predicted == test_y).sum().item()
        total += test_y.size(0)

        # Compute class-wise accuracy
        for i in range(10):
            class_total[i] += (test_y == i).sum().item()
            class_correct[i] += (predicted == i)[test_y == i].sum().item()

    overall_accuracy = correct / total
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

    # Print class-wise accuracy
    for i in range(10):
        class_accuracy = class_correct[i] / class_total[i]
        print(f"Class {i} Accuracy: {class_accuracy:.4f}")

model.train() 

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), list(loss_at_epoch.values()), marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.grid(True)
plt.savefig('loss_plot.png')

