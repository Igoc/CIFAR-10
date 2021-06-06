import torch
import torch.nn            as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(inputChannels, outputChannels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(outputChannels)
        
        self.conv2 = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(outputChannels)
        
        self.conv3 = nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(outputChannels)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.pool(x)
        
        return x

class Network(nn.Module):
    def __init__(self, classNumber=10):
        super(Network, self).__init__()
        
        self.block1 = Block(inputChannels=3, outputChannels=32)
        self.block2 = Block(inputChannels=32, outputChannels=64)
        self.block3 = Block(inputChannels=64, outputChannels=96)
        
        self.fc1 = nn.Linear(in_features=96 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=classNumber)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    
    print(f'[Device] {device}\n')
    
    model     = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
    
    print(f'[Model]\n{model}\n')
    
    inputs  = torch.randn((2, 3, 32, 32)).to(device)
    targets = torch.zeros(2, dtype=torch.int64).to(device)
    
    print(f'[Dummy inputs]\n{inputs}\n')
    print(f'[Dummy targets]\n{targets}\n')
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss    = criterion(outputs, targets)
    
    print(f'[Dummy outputs]\n{outputs.argmax(dim=1)}')
    
    loss.backward()
    optimizer.step()