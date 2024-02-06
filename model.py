import torch
import torch.nn as nn

# Define the NeuralNet class
class NeuralNet(nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Define the layers
        self.l1 = nn.Linear(input_size, hidden_size)  
        self.l2 = nn.Linear(hidden_size, hidden_size)  
        self.l3 = nn.Linear(hidden_size, num_classes)  
        self.relu = nn.ReLU()  
    # Forward method
    def forward(self, x):
        out = self.l1(x)  # Pass input through first layer
        out = self.relu(out)  
        out = self.l2(out)  
        out = self.relu(out)  
        out = self.l3(out)  
        
        return out
