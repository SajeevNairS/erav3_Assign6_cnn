from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from itertools import accumulate
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2, 2)
        # Second block
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        
        # Final block
        self.conv6 = nn.Conv2d(16, 10, kernel_size=1)  # 1x1 conv to reduce channels
        
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn3(self.conv3(
            F.relu(self.bn2(self.conv2(
            F.relu(self.bn1(self.conv1(x))))))))))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second block
        F.relu(self.bn5(self.conv5(
            F.relu(self.bn4(self.conv4(x))))))
        x = self.dropout(x)
        
        # Final block
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
calculate_receptive_field()

# Calculate and print receptive field for each layer
def calculate_receptive_field():
    rf = 1  # Initial receptive field
    stride = 1  # Cumulative stride
    print("\nReceptive Field Analysis:")
    print("Layer\t\tRF\tStride\tOutput Size")
    print("-" * 50)
    
    # First block
    rf = rf + 2 * (3-1)  # conv1 (3x3)
    print("conv1\t\t{}x{}\t{}\t28x28".format(rf, rf, stride))
    
    rf = rf + 2 * (3-1)  # conv2 (3x3)
    print("conv2\t\t{}x{}\t{}\t28x28".format(rf, rf, stride))
    
    rf = rf + 2 * (3-1)  # conv3 (3x3)
    print("conv3\t\t{}x{}\t{}\t28x28".format(rf, rf, stride))
    
    rf, stride = rf * 2, stride * 2  # First pool
    print("pool1\t\t{}x{}\t{}\t14x14".format(rf, rf, stride))
    
    rf, stride = rf * 2, stride * 2  # Second pool
    print("pool2\t\t{}x{}\t{}\t7x7".format(rf, rf, stride))
    
    rf = rf + 2 * (3-1) * stride  # conv4 (3x3)
    print("conv4\t\t{}x{}\t{}\t7x7".format(rf, rf, stride))
    
    rf = rf + 2 * (3-1) * stride  # conv5 (3x3)
    print("conv5\t\t{}x{}\t{}\t7x7".format(rf, rf, stride))
    
    rf = rf + 2 * (1-1) * stride  # conv6 (1x1)
    print("conv6\t\t{}x{}\t{}\t7x7".format(rf, rf, stride))
    
    print("\nFinal receptive field: {}x{}".format(rf, rf))

# Place this right after the model summary
print("\nReceptive Field Analysis:")
calculate_receptive_field()


torch.manual_seed(1)
batch_size = 128
#batch_size = 16

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), 
                                            scale=(0.85, 1.15), shear=(-10, 10)),
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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    div_factor=25,
    final_div_factor=1000,
)
accuracy = 0.0

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    accuracy = test(model, device, test_loader)
    if accuracy > 99.4:
        print("Accuracy is greater than 99.4 and reach at", epoch)
        #break


