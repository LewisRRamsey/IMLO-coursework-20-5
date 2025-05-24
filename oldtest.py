import torch
import torchvision
import torchvision.transforms as transforms
import train



test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

test_data_set = torchvision.datasets.CIFAR10(root='./testdata', train=False, download=True, transform=test_transform)

test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

my_neural_network = train.NeuralNetwork()
my_neural_network.load_model('./pytorch_trained_model.pth')

with torch.no_grad():
    correct_classifications = 0
    total_classifications = 0
    for test_data in test_data_loader:
        test_images, test_labels = test_data
        classification_outputs = my_neural_network.forward(test_images)
        predicted = classification_outputs.argmax(dim=1)
        total_classifications += test_labels.size(0)
        correct_classifications += (predicted == test_labels).sum().item()

    print('Test accuracy:', 100 * correct_classifications // total_classifications, '%')