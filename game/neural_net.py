import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class LinearNeuralNet(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearNeuralNet, self).__init__()
        input_size = 3*board_size**2
        self.l1 = nn.Linear(input_size, input_size*8)
        self.l2 = nn.Linear(input_size*8, input_size*16)
        self.l3 = nn.Linear(input_size*16, input_size*16)
        self.l4 = nn.Linear(input_size*16, input_size*8)
        self.l5 = nn.Linear(input_size*8, board_size**2)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.flatten(x, start_dim=1)
        
        x = self.l1(x)
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


        x = self.l5(x)
        x = F.softmax(x, dim=1)
        return x

class LinearResidualNet(torch.nn.Module):
    def __init__(self, board_size: int):
        super(LinearResidualNet, self).__init__()
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

# class LinearNeuralNet(torch.nn.Module):
#     def __init__(self, board_size: int):
#         super(LinearNeuralNet, self).__init__()
#         input_size = 2*board_size**2
#         self.l1 = nn.Linear(2*board_size**2, input_size*4)
#         self.l2 = nn.Linear(input_size*4, input_size*8)
#         self.l3 = nn.Linear(input_size*8, input_size*4)
#         self.l4 = nn.Linear(input_size*4, input_size//2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         x = torch.flatten(x, start_dim=1)
        
#         x = self.l1(x)
#         x = F.relu(x)

#         x = self.l2(x)
#         x = F.relu(x)

#         x = self.l3(x)
#         x = F.relu(x)

#         x = self.l4(x)
#         x = F.softmax(x, dim=1)
#         return x
    
class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, board_size):
        super(ConvolutionalNeuralNet, self).__init__()
        self.board_size = board_size
    
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1) # x = x.view(-1, 128 * self.board_size * self.board_size)

        # Fully connected layer
        x = self.fc(x)

        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)

        return x
    
class DeepConvolutionalNeuralNet(nn.Module):
    def __init__(self, board_size):
        super(DeepConvolutionalNeuralNet, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected layer
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)
        # may be necessary to add more FC layers, testing will show

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten the tensor
        x = x.view(-1, 128 * self.board_size * self.board_size)

        # Fully connected layer
        x = self.fc(x)

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