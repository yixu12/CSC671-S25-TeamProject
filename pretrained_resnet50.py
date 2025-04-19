"""
This script uses:
- PyTorch [Paszke et al., 2019]: https://arxiv.org/abs/1912.01703
- Torchvision: https://github.com/pytorch/vision
- ResNet-50 model [He et al., 2016]: https://arxiv.org/abs/1512.03385
"""

# This script is for training a PyTorch ResNet-50 implementation, on a skin cancer dataset.
# Check https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
# Change the model and hyperparameters as needed, or run this in a Jupyter notebook.


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import time


def main():
    # Data path
    data_dir = "archive/Skin cancer ISIC The International Skin Imaging Collaboration/Train" # or wherever your data is located

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet1K stats
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=data_dir.replace("Train", "Test"), transform=transform_test)

    print(f"CPU count: {os.cpu_count()}")
    num_workers = min(4, os.cpu_count() // 2)  # Dynamically set num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers) # adjust batch_size to fit your GPU memory
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=num_workers)   # currently 64 uses ~7GB on RTX 3080

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Load the model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists("test_weights.pth"):  # check for existing weights
        checkpoint = torch.load("test_weights.pth", map_location=device)
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Classes: {train_dataset.classes}")

    # Train the model
    model.to(device)
    print(f"Using {next(model.parameters()).device}")

    # ----------NO MIXED PRECISION----------
    num_epochs = 3  # Set the number of training epochs
    start = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {running_loss / len(train_loader):.4f}")

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Print the total training time
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the model and optimizer state
    save_path = "test_weights.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print(f"Model and optimizer state saved to {save_path}")
    # ----------NO MIXED PRECISION----------



    # #----------MIXED PRECISION----------
    # # Generate Warnings
    # # Reduces both Average Loss AND Validation Accuracy? (Need more epochs to test)
    # scaler = torch.cuda.amp.GradScaler()
    #
    # num_epochs = 3
    # start = time.time()
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #
    #         with torch.cuda.amp.autocast():
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #
    #         optimizer.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #
    #         running_loss += loss.item()
    #     print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {running_loss / len(train_loader):.4f}")
    #
    #     # Validate the model
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             _, predicted = torch.max(outputs, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #     print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    #
    # # Print the total training time
    # end = time.time()
    # print(f"Training completed in {end - start:.2f} seconds")
    #
    # # Save the model and optimizer state
    # save_path = "test_weights.pth"
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, save_path)
    # print(f"Model and optimizer state saved to {save_path}")
    # # ----------MIXED PRECISION----------

if __name__ == '__main__':
    main()