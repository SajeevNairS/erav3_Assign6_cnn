from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from itertools import accumulate
from torchsummary import summary
from model import create_model  # Import the model from the new file

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print("\nModel Parameters:", count_parameters(model))
    print("Model Architecture:")
    print(model)

    # Calculate receptive field
    rf = 1  # Initial receptive field
    stride = 1  # Cumulative stride
    input_size = 28  # Starting input size
    print("\nReceptive Field Analysis:")
    print("Layer\t\tRF\tStride\tOutput Size")
    print("-" * 50)
    
    # Helper function to calculate output size for conv layer
    def get_output_size(input_size, kernel_size, stride, padding):
        return ((input_size + 2*padding - kernel_size) // stride) + 1
    
    # Analyze each layer in sequence
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # Update RF for conv layer
            if layer.kernel_size[0] > 1:  # Skip 1x1 convolutions for RF calculation
                rf = rf + 2 * (layer.kernel_size[0]-1) * stride
            if layer.stride[0] > 1:
                stride *= layer.stride[0]
            input_size = get_output_size(input_size, layer.kernel_size[0], layer.stride[0], layer.padding[0])
            print(f"{name}\t\t{rf}x{rf}\t{stride}\t{input_size}x{input_size}")
            
        elif isinstance(layer, nn.MaxPool2d):
            # Update RF and stride for pooling layer
            rf *= layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            stride *= layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            input_size = get_output_size(input_size, 
                                       layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
                                       layer.stride if isinstance(layer.stride, int) else layer.stride[0],
                                       0)  # MaxPool typically doesn't have padding
            print(f"{name}\t\t{rf}x{rf}\t{stride}\t{input_size}x{input_size}")
            
        elif isinstance(layer, nn.AvgPool2d):
            input_size = 1  # GAP reduces spatial dimensions to 1x1
            print(f"{name}\t\t{rf}x{rf}\t{stride}\t{input_size}x{input_size}")
    
    print("\nFinal receptive field: {}x{}".format(rf, rf))

# Remove torchsummary import and replace device setup
device = get_device()
print(f"Using device: {device}")

# Create model and move to device
model = create_model().to(device)
print_model_summary(model)

# Update kwargs based on device
kwargs = {}
if device.type == "cuda":
    kwargs = {'num_workers': 4, 'pin_memory': True}
elif device.type == "mps":
    kwargs = {'num_workers': 0}  # MPS works better with 0 workers
else:
    kwargs = {'num_workers': 2}

torch.manual_seed(1)
batch_size = 128
#batch_size = 16

dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       #transforms.RandomRotation((-7.0, 7.0),fill=(0,)),
                       transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1),shear=(-5, 5)),
                       transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        pbar.set_description(desc= f'Epoch : {epoch} loss={loss.item()} accuracy={accuracy} batch_id={batch_idx}')





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

model = create_model().to(device)
if device.type != "mps" :
    summary(model, input_size=(1, 28, 28))

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=10e-4, momentum=0,dampening=0,weight_decay=0,nesterov=False)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
accuracy = 0.0

# Initialize finalprint and max_accuracy before the training loop
finalprint = ""
max_accuracy = 0.0

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    accuracy = test(model, device, test_loader)
    
    # Update max_accuracy and finalprint if current accuracy is higher
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        finalprint = f"Maximum Accuracy achieved: {max_accuracy:.2f}% at epoch {epoch}"
    
    if accuracy >= 99.4:
        print("Accuracy is greater than 99.4 and reached at epoch", epoch)

print(f"{finalprint}")

def main():
    # Your main training/testing loop
    model = create_model()  # Use the imported create_model function
    # Rest of your code remains the same
    pass

if __name__ == '__main__':
    main()