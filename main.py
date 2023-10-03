import torch
import torch.nn as nn
import torch.optim as optim
import network
import dataset
import numpy as np

num_epochs = 100
batch_size = 5
learning_rate = 0.001 

trainloader = torch.utils.data.DataLoader(dataset=dataset.My_Dataset(), batch_size=batch_size, shuffle=True)


def train(model, trainloader, optimizer, criterion, epoch):
    model.train() 
    running_loss = 0.0 
    for i, data in enumerate(trainloader, 0): 
        inputs, labels = data 
        labels = labels.unsqueeze(1)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        labels = labels.float()
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() 
        if i % 14 == 13:    
            print('[%d, %5d] loss: %.20f' % (epoch + 1, i + 1, running_loss / 14.0))
            running_loss = 0.0

if __name__ == '__main__':
    
    model = network.CNN()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs): 
        train(model, trainloader, optimizer, criterion, epoch) 
    para = network.CNN.output()
    torch.save(para, "para.pt")
