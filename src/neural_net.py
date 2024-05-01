import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import *


class LinearNeuralNet(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearNeuralNet, self).__init__()
        input_size = 3*board_size**2

        if ACTIVATION_FUNCTION == 'relu':
            activation = nn.ReLU()
        elif ACTIVATION_FUNCTION == 'sigmoid':
            activation = nn.Sigmoid()
        elif ACTIVATION_FUNCTION == 'tanh':
            activation = nn.Tanh()
        elif ACTIVATION_FUNCTION == 'linear':
            activation = nn.Linear()
        else:
            raise ValueError(f'Invalid activation function: {ACTIVATION_FUNCTION}')
        

        dropout = nn.Dropout(DROPOUT_PROB)

        self.num_hidden_layers = len(NUM_NEURONS) - 1

        self.sequential = nn.Sequential(
            nn.Linear(input_size, input_size*NUM_NEURONS[0]),
            activation,
            dropout
            )
        
        for i in range(self.num_hidden_layers):
            self.sequential.append(nn.Linear(input_size*NUM_NEURONS[i], input_size*NUM_NEURONS[i+1]))
            self.sequential.append(activation)
            self.sequential.append(dropout)

        self.sequential.append(nn.Linear(input_size*NUM_NEURONS[self.num_hidden_layers], board_size**2))
        self.sequential.append(activation)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)

        x = self.sequential(x)

        x = F.softmax(x, dim=1)
        return x
    

class LinearResidualNet(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearResidualNet, self).__init__()
        input_size = 3*board_size**2

        if ACTIVATION_FUNCTION == 'relu':
            activation = nn.ReLU()
        elif ACTIVATION_FUNCTION == 'sigmoid':
            activation = nn.Sigmoid()
        elif ACTIVATION_FUNCTION == 'tanh':
            activation = nn.Tanh()
        elif ACTIVATION_FUNCTION == 'linear':
            activation = nn.Linear()
        else:
            raise ValueError(f'Invalid activation function: {ACTIVATION_FUNCTION}')
        

        dropout = nn.Dropout(DROPOUT_PROB)

        self.num_hidden_layers = len(NUM_NEURONS) - 1

        self.sequential = nn.Sequential(
            nn.Linear(input_size, input_size*NUM_NEURONS[0]),
            activation,
            dropout
            )
        
        for i in range(self.num_hidden_layers):
            self.sequential.append(nn.Linear(input_size*NUM_NEURONS[i], input_size*NUM_NEURONS[i+1]))
            self.sequential.append(activation)
            self.sequential.append(dropout)

        self.final_layer = nn.Sequential(
            nn.Linear(input_size*NUM_NEURONS[self.num_hidden_layers]+board_size**2, board_size**2),
            activation
            )
        
    
    def forward(self, input_matrix: torch.Tensor) -> torch.Tensor:
        occupied = 1 - input_matrix[:,-1,:,:]
        occupied = torch.flatten(occupied, start_dim=1)
        x = torch.flatten(input_matrix, start_dim=1)

        x = self.sequential(x)

        # Merge the input with the network output
        # This way the last layer can easily learn to set positions that are occupied to 0
        x = torch.hstack((x, occupied))

        x = self.final_layer(x)

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
    

class ConvolutionalNeuralNetOld(nn.Module):
    def __init__(self, board_size, channels = 128, layers = 8):
        super(ConvolutionalNeuralNetOld, self).__init__()
        self.board_size = board_size
        self.layers_middle = layers - 1
        in_channels = 3

        self.conv_start = torch.nn.Conv2d(in_channels, channels, kernel_size=5, padding=0)

        # This is a BAD idea, use nn.Sequential instead
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
    
    def to(self, device):
        self.conv_start.to(device)
        for i in range(self.layers_middle):
            self.conv_middle[i].to(device)
        self.conv_end.to(device)


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
    