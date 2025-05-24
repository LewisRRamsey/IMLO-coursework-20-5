import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


# 0.5 is the mean of the cifar training set values in a tensor
cifar_data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data_set = torchvision.datasets.CIFAR10(root='./trainingdata', train=True, download=True, transform=cifar_data_transform)

training_data_loader = torch.utils.data.DataLoader(training_data_set, batch_size=256, shuffle=True, num_workers=2)

batch_size = training_data_loader.batch_size


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 sets of 32*32 layers per image due to rgb colouring
        # padding used to capture the edges of the image
        # kernel size of 4 used to reduce overfitting
        self.convolution_layer1 = nn.Conv2d(3, 32, 4, 1, padding = 2)
        self.convolution_layer2 = nn.Conv2d(32, 64, 4, 1, padding = 2)
        self.convolution_layer3 = nn.Conv2d(64, 128, 4, 1, padding = 2)

        # after 3 max pooling layers the image is reduced to 4*4
        # 128 is the number of output channels from the last convolution layer
        self.fully_connected_layer = nn.Linear(4*4*128, 1000)

        # dropout used to prevent co-adaption between neurons
        # 1000 neurons used in the fully connected layer so higher dropout rate needed
        self.dropout = nn.Dropout(0.325)

        # converts to ouput features, 10 classes for cifar10
        self.output_layer = nn.Linear(1000, 10)


    def forward(self, data_input):

        # F.relu is the activation function used at each convolution layer (outputs input if positive, else outputs 0)
        # applying convolutions to the image and max pooling after each convolution (reduces dimensionality of each image)
        data = F.relu(self.convolution_layer1(data_input))
        data = F.max_pool2d(data, 2, 2)
        data = F.relu(self.convolution_layer2(data))
        data = F.max_pool2d(data, 2, 2)
        data = F.relu(self.convolution_layer3(data))
        data = F.max_pool2d(data, 2, 2)

        # flattening the image to a 1D vector, so can be passed to the fully connected layers
        # 4*4*128 is the output size of the last convolution layer
        data = data.view(-1, 4*4*128)


        data = F.relu(self.fully_connected_layer(data))
        data = self.dropout(data)
        data_output = self.output_layer(data)

        return data_output

    def save_model(self, path):
        # saves the model's state dictionary to the specified path
        torch.save(self.state_dict(), path)
        print('Model saved to:', path)

    def load_model(self, path):
        # loads the model's state dictionary from the specified path
        self.load_state_dict(torch.load(path))
        print('Model loaded from:', path)

    def train_model(self, num_epochs):

        # epochs are how many times data set is iterated over to train the model
        for epoch in range(num_epochs):

            # running loss and running correct predictions are used to calculate the average loss and accuracy for each epoch
            running_loss_value = 0.0
            running_correct_predictions = 0.0

            # iterating over the training data loader (batch by batch)
            for training_input, training_label in training_data_loader:

                # passing data as input through the neural network
                training_output = self(training_input)

                batch_loss = log_loss(training_output, training_label)

                # zeroing the gradients of the optimiser
                stochastic_optimiser.zero_grad()

                # beginning of backpropagation (computing gradients of neural network parameters with respect to the loss function)
                batch_loss.backward()

                # gradients used to update the parameters of the neural network
                stochastic_optimiser.step()

                _, predicted_class = torch.max(training_output, 1)

                running_loss_value += batch_loss.item()

                running_correct_predictions += torch.sum(predicted_class == training_label.data)

            epoch_loss = running_loss_value / (len(training_data_loader) * batch_size)

            epoch_accuracy = running_correct_predictions.float() / len(training_data_set)

            # returns loss and accuracy for each epoch
            print('At epoch', epoch + 1, 'loss:', epoch_loss, 'accuracy:', epoch_accuracy.item() * 100)



# create instance of the neural network
my_neural_network = NeuralNetwork()

# define loss function to be used in training of model
log_loss = nn.CrossEntropyLoss()

# define the optimiser to be used in training of model
# using stochastic gradient descent
stochastic_optimiser = torch.optim.SGD(my_neural_network.parameters(), lr=0.005, momentum=0.9)


# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':
    my_neural_network.train_model(num_epochs = 40)
    my_neural_network.save_model('./pytorch_trained_model.pth')
    print('Training is complete')