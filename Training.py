from Model.Network            import Network

from tqdm                     import tqdm

import torch
import torch.nn               as nn
import torch.optim            as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os

DATA_PATH       = './Data'
CHECKPOINT_PATH = './Checkpoint'

bestAccuracy = 0

def parseArguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='enables verbosity')
    parser.add_argument('--checkpoint', action='store_true', default=False, help='enables checkpoint')
    parser.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--batchsize', type=int, default=64, metavar='N', help='input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    
    return parser.parse_args()

def train(model, device, criterion, optimizer, dataLoader, epoch):
    model.train()
    
    with tqdm(dataLoader, unit=' batch') as progress:
        for inputs, targets in progress:
            progress.set_description(f'Epoch {epoch}')
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs  = model(inputs)
            accuracy = (outputs.argmax(dim=1) == targets).sum().item() / len(inputs) * 100
            loss     = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            progress.set_postfix(accuracy=accuracy, loss=loss.item())

def test(model, device, criterion, dataLoader, checkpoint):
    global bestAccuracy
    
    model.eval()
    
    correct     = 0
    averageLoss = 0
    
    with torch.no_grad():
        for inputs, targets in dataLoader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            correct     += (outputs.argmax(dim=1) == targets).sum().item()
            averageLoss += criterion(outputs, targets).item()
    
    accuracy     = correct / len(dataLoader.dataset) * 100
    averageLoss /= len(dataLoader.dataset)
    
    print(f'[Test] accuracy={accuracy:.2f} ({correct}/{len(dataLoader.dataset)}), average loss={averageLoss:.4f}', flush=True)
    
    if checkpoint == True and accuracy > bestAccuracy:
        if os.path.isdir(CHECKPOINT_PATH) == False:
            os.makedirs(CHECKPOINT_PATH)
        
        torch.save(model.state_dict(), f'{CHECKPOINT_PATH}/model.pth')
        
        print(f'[Checkpoint] Improving accuracy ({bestAccuracy:.2f} -> {accuracy:.2f})\n', flush=True)
        
        bestAccuracy = accuracy
    else:
        print(flush=True)

if __name__ == '__main__':
    arguments = parseArguments()
    
    trainingTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainingDataset    = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=trainingTransform)
    trainingDataLoader = torch.utils.data.DataLoader(trainingDataset, batch_size=arguments.batchsize, shuffle=True, num_workers=0)
    
    testDataset    = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=testTransform)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=arguments.batchsize, shuffle=False, num_workers=0)
    
    torch.manual_seed(arguments.seed)
    
    cuda   = not arguments.cpu and torch.cuda.is_available()
    device = 'cuda' if cuda == True else 'cpu'
    
    if arguments.verbose == True:
        print(f'\n[Device] {device}\n', flush=True)
    
    model     = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=arguments.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    trainingCriterion = nn.CrossEntropyLoss()
    testCriterion     = nn.CrossEntropyLoss(reduction='sum')
    
    if arguments.verbose == True:
        print(f'[Model]\n{model}\n', flush=True)
    else:
        print(flush=True)
    
    for epoch in range(1, arguments.epochs + 1):
        train(model, device, trainingCriterion, optimizer, trainingDataLoader, epoch)
        test(model, device, testCriterion, testDataLoader, arguments.checkpoint)
        
        scheduler.step()