import torch
import torch.nn as nn

batch_size = 64 
num_epochs = 10 
learning_rate = 0.01 

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FCNN, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        
        self.a1 = nn.ReLU()
        self.a2 = nn.Tanh()
    def forward(self, x):
        out = self.linear1(x) 
        out = self.a1(out) 
        out = self.linear2(out)
        out = self.a2(out)
        out = self.linear3(out)
        return out
    
class CNN(nn.Module):
    global x_t
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, 1, 1) 
        self.pool = nn.MaxPool1d(2, 2) 
        self.conv2 = nn.Conv1d(16, 32, 3, 1, 1) 

        self.fc1 = nn.Linear(32*16000, 128) 
        self.fc2 = nn.Linear(128, 96) 
        self.fc3 = nn.Linear(96, 1) 
        self.a1 = nn.ReLU()
        self.a2 = nn.ReLU()
        self.a3 = nn.Sigmoid()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32*16000) 
        global x_t
        x_t = x
        x = self.a1(self.fc1(x)) 
        x = self.a2(self.fc2(x)) 
        x = self.a3(self.fc3(x)) 
        return x
    
    def output():
        global x_t
        return x_t
