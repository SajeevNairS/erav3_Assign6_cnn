import torch
from assign6 import Net, test
from torchvision import datasets, transforms

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Net().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    
    # Test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True)
    
    # Run test
    accuracy = test(model, device, test_loader)
    print(f"Final Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 