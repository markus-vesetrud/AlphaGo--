import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import *


class LinearResidualNet(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearResidualNet, self).__init__()
        input_size = 3*board_size**2

        self.num_hidden_layers = len(NUM_NEURONS) - 1

        self.layers = [None] * (self.num_hidden_layers)
        self.layer_in = nn.Linear(input_size, input_size*NUM_NEURONS[0])
        
        for i in range(self.num_hidden_layers):
            self.layers[i] = nn.Linear(input_size*NUM_NEURONS[i], input_size*NUM_NEURONS[i+1])

        self.layer_out = nn.Linear(input_size*NUM_NEURONS[self.num_hidden_layers]+board_size**2, board_size**2)
        self.dropout = nn.Dropout(0.25)

        if ACTIVATION_FUNCTION == 'relu':
            self.activation = F.relu
        elif ACTIVATION_FUNCTION == 'sigmoid':
            self.activation = F.sigmoid
        elif ACTIVATION_FUNCTION == 'tanh':
            self.activation = F.tanh
        elif ACTIVATION_FUNCTION == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError(f'Invalid activation function: {ACTIVATION_FUNCTION}')
    
    def to(self, device):
        self.layer_in.to(device)
        for i in range(self.num_hidden_layers):
            self.layers[i].to(device)
        self.layer_out.to(device)
        self.dropout.to(device)
        return self

    def forward(self, input_matrix: torch.Tensor) -> torch.Tensor:
        occupied = 1 - input_matrix[:,-1,:,:]
        occupied = torch.flatten(occupied, start_dim=1)
        input_matrix = torch.flatten(input_matrix, start_dim=1)
        
        x = self.layer_in(input_matrix)
        x = self.activation(x)
        x = self.dropout(x)

        for i in range(self.num_hidden_layers):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        # Merge the input with the network output
        # This way the last layer can easily learn to set positions that are occupied to 0
        x = torch.hstack((x, occupied))

        x = self.layer_out(x)
        x = F.softmax(x, dim=1)
        return x


class LinearResidualNetOld(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearResidualNetOld, self).__init__()
        input_size = 3*board_size**2
        self.l1 = nn.Linear(input_size, input_size*8)
        self.l2 = nn.Linear(input_size*8, input_size*16)
        self.l3 = nn.Linear(input_size*16, input_size*16)
        self.l4 = nn.Linear(input_size*16, input_size*8)
        self.l5 = nn.Linear(input_size*8+board_size**2, board_size**2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input_matrix: torch.Tensor) -> torch.Tensor:
        occupied = 1 - input_matrix[:,-1,:,:]
        occupied = torch.flatten(occupied, start_dim=1)
        input_matrix = torch.flatten(input_matrix, start_dim=1)
        
        x = self.l1(input_matrix)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.l2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.l3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.l4(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Merge the input with the network output
        # This way the last layer can easily learn to set positions that are occupied to 0
        x = torch.hstack((x, occupied))

        x = self.l5(x)
        x = F.softmax(x, dim=1)
        return x
    

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, board_size, channels = 128, layers = 8):
        super(ConvolutionalNeuralNet, self).__init__()
        self.board_size = board_size
        self.layers_middle = layers - 1
        in_channels = 3

        self.conv_start = torch.nn.Conv2d(in_channels, channels, kernel_size=5, padding=0)

        self.conv_middle = []
        for _ in range(self.layers_middle):
            self.conv_middle.append(torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        
        # Connects all the channels per board position to a single board position
        self.conv_end = torch.nn.Conv2d(channels, 1, kernel_size=1, padding=0)

        if ACTIVATION_FUNCTION == 'relu':
            self.activation = F.relu
        elif ACTIVATION_FUNCTION == 'sigmoid':
            self.activation = F.sigmoid
        elif ACTIVATION_FUNCTION == 'tanh':
            self.activation = F.tanh
        elif ACTIVATION_FUNCTION == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError(f'Invalid activation function: {ACTIVATION_FUNCTION}')


    def add_padding(self, board: torch.Tensor) -> torch.Tensor:
        # add a padding of 2 around the board
        black_channel = board[:,0,:,:]
        red_channel = board[:,1,:,:]
        empty_channel = board[:,2,:,:]
        black_channel = F.pad(black_channel, (0, 0, 2, 2, 0, 0), mode='constant', value=0.0)
        black_channel = F.pad(black_channel, (2, 2, 0, 0, 0, 0), mode='constant', value=1.0)
        red_channel = F.pad(red_channel, (0, 0, 2, 2, 0, 0), mode='constant', value=1.0)
        red_channel = F.pad(red_channel, (2, 2, 0, 0, 0, 0), mode='constant', value=0.0)
        empty_channel = F.pad(empty_channel, (2, 2, 2, 2, 0, 0), mode='constant', value=0.0)

        board = torch.stack((black_channel, red_channel, empty_channel), dim=1)
        
        return board 

    def forward(self, x):
        x = self.add_padding(x)
        
        x = self.conv_start(x)
        x = self.activation(x)

        for i in range(self.layers_middle):
            x = self.conv_middle[i](x)
            x = self.activation(x)

        x = self.conv_end(x)
        x = self.activation(x)

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)

        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)

        return x
    

# --------------------------------
    
# Example of usage

""" 
# Hyperparameters
input_size = 28*28
hidden_size = 500
output_size = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=False, 
                                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size, 
                                             shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

# Model
model = LinearNeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt') """