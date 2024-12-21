from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
from itertools import accumulate

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  # Stabilize learning
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First spatial reduction

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(10) 
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,stride=2)
        self.bn6 = nn.BatchNorm2d(16)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second spatial reduction

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(20)
        self.conv8 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(10)

        self.gap = nn.AvgPool2d(kernel_size=2)  # Global average pooling
        self.conv9 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(0.05)  # Light regularization
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.05)
        self.dropout3 = nn.Dropout(0.05)
        self.dropout4 = nn.Dropout(0.05)
        self.dropout5 = nn.Dropout(0.05)
        self.dropout6 = nn.Dropout(0.05)
        self.dropout7 = nn.Dropout(0.05)
        self.dropout8 = nn.Dropout(0.05)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool1(x)
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        x = self.pool2(x)
        
        x = self.dropout7(F.relu(self.bn7(self.conv7(x))))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.gap(x)
        self.conv9(x)

        #print(x.shape)
        
        x = x.view(x.size(0), -1)
        
        return F.log_softmax(x, dim=-1)



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))



torch.manual_seed(1)
batch_size = 128
#batch_size = 16

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       #transforms.RandomRotation((-7.0, 7.0),fill=(1,)),
                       transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                       #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))




test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
   batch_size=batch_size, shuffle=True, **kwargs)

#train1_loader, test1_loader = torch.utils.data.random_split(dataset, [50000, 10000])

train_loader = torch.utils.data.DataLoader(dataset,
   batch_size=batch_size, shuffle=True, **kwargs)

#test_loader = torch.utils.data.DataLoader(test_d_loader,batch_size=batch_size, shuffle=True, **kwargs)







def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    accuracy = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).float().sum().item()
        accuracy = float(100. * correct / len(train_loader.dataset))
        pbar.set_description(desc= f'Epoch : {epoch} dataset size={len(train_loader.dataset)} loss={loss.item()} accuracy={accuracy} batch_id={batch_idx}')





def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).float().sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = float(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        float(100. * (correct / len(test_loader.dataset)))))
    return accuracy

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        div_factor=10,
        final_div_factor=100,
        #anneal_strategy='cos'
    )
accuracy = 0.0

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    accuracy = test(model, device, test_loader)
    if accuracy > 99.4:
        print("Accuracy is greater than 99.4 and reach at", epoch)
        #break


