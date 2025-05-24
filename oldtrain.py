import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


training_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

training_data_set = torchvision.datasets.CIFAR10(root='./traindata', train=True, download=True, transform=training_transform)

training_data_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, shuffle=True)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnmodel = nn.Sequential(
            nn.Conv2d(3, 6, 5), # Convolution layer
            nn.ReLU(),          # Activation function
            nn.AvgPool2d(2, 2), # Pooling layer
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.fullyconn = nn.Sequential(
            nn.Linear(400, 120), # Linear Layer
            nn.ReLU(),           # ReLU Activation Function
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnnmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fullyconn(x)
        return x
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print('Model saved to', path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print('Model loaded from', path)

my_neural_network = NeuralNetwork()

cross_entropy_loss = nn.CrossEntropyLoss()

# optimiser using Adam with learning rate 0.001
optimizer = torch.optim.Adam(my_neural_network.parameters(), lr=0.001)

if __name__ == '__main__':

    for epoch in range(20):

        running_loss = 0.0
        for epoch_index, data in enumerate(training_data_loader, 0):

            inputs, labels = data

            outputs = my_neural_network.forward(inputs)

            # computes the loss using the cross entropy loss function (log loss)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()

            optimizer.step()

            # resets the gradients for the gradient descent (back to zero)
            optimizer.zero_grad()

            running_loss += loss.item()

            if epoch_index % 5000 == 4999: 
                print('At epoch', epoch + 1, 'iteration', epoch_index + 1, 'loss:', running_loss / 5000)
                running_loss = 0.0

    print('Training is complete')

# Saving the model to a file for reuse on testing
my_neural_network.save_model('./pytorch_trained_model.pth')