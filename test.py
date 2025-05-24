import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import train

#transform for test data same as training data
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 256

# CIFAR10 test dataset
test_data_set = torchvision.datasets.CIFAR10(root='./testdata', train=False, download=True, transform=test_transform)

test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

# Load the trained neural network model
my_neural_network = train.NeuralNetwork()
my_neural_network.load_model('./pytorch_trained_model.pth')
my_neural_network.eval()

#specify the same loss function as used in training
log_loss = nn.CrossEntropyLoss()

# Evaluate the model on the test dataset, gradients not needed for evaluation
with torch.no_grad():

    correct_test_classifications = 0

    # Iterate through the test data loader
    for test_input, test_label in test_data_loader:

        test_outputs = my_neural_network(test_input)

        batch_loss = log_loss(test_outputs, test_label)

        _, test_prediction_class = torch.max(test_outputs, 1)

        correct_test_classifications += torch.sum(test_prediction_class == test_label.data)

    test_accuracy = correct_test_classifications.float() / len(test_data_set)
    print('Test accuracy:', test_accuracy.item() * 100, '%')



        

