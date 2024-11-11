import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
data_dir = '/scratch/sanskar/NewR22/'  # Replace with the actual path to your dataset folder

# Data Augmentation and Normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load Data
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# Load Pre-trained ResNet Model
#model = models.resnet18(pretrained=True)
weights = ResNet18_Weights.IMAGENET1K_V1  
#weights = ResNet18_Weights.DEFAULT  

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training Function
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

# Train the Model
model = train_model(model, criterion, optimizer, num_epochs=10)

# Testing and Displaying Results
def visualize_model(model, num_images=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 10))

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size()[0]):
                is_correct = preds[i] == labels[i]
                
                # Display only a specified number of images
                if images_so_far == num_images:
                    return

                # Set title based on correct or incorrect prediction
                title = f"Predicted: {class_names[preds[i]]} (Correct)" if is_correct else f"Predicted: {class_names[preds[i]]} (Incorrect)"

                # Show only a balanced mix of correct and incorrect predictions
                if is_correct and images_so_far % 2 == 0:
                    images_so_far += 1
                elif not is_correct and images_so_far % 2 != 0:
                    images_so_far += 1
                else:
                    continue

                # Plot image
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(title)
                img = inputs.cpu().data[i].numpy().transpose((1, 2, 0))
                img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
                plt.imshow(img)

    plt.tight_layout()
    plt.show()

# Visualize Correct and Incorrect Predictions
visualize_model(model,num_images=10)

