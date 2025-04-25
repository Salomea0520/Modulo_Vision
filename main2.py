from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot as plt
import numpy as np

# Create a transform function for Skin Cancer dataset
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the dataset from the local directory
train_dir = '/home/22061506mercedes/Computer_Vision1_v3/skin_cancer_dataset/train-20240324T151905Z-001/train'
test_dir = '/home/22061506mercedes/Computer_Vision1_v3/skin_cancer_dataset/test-20240324T151902Z-001/test'
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_dataset

train_dataset.classes, train_dataset.class_to_idx

type(train_dataset)

test_dataset

test_dataset.classes, test_dataset.class_to_idx

type(test_dataset)

# Vamos a utilizar ResizeCrop y HorizontalFlip para aumentar el dataset train
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset from the local directory with data augmentation
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)

train_dataset

train_dataset.classes, train_dataset.class_to_idx

type(train_dataset)

# Vamos a utilizar ResizeCrop y HorizontalFlip para aumentar el dataset test
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset from the local directory with data augmentation
test_dataset = datasets.ImageFolder(root=train_dir, transform=transform_test)


test_dataset

test_dataset.classes, train_dataset.class_to_idx

type(test_dataset)

# Vamos a definir el tamaño del batch y el número de workers
batch_size = 32
num_workers = 4
# Create data loaders for training and testing datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Check the number of batches in the training and testing datasets
print(f"Number of batches in training dataset: {len(train_loader)}")
print(f"Number of batches in testing dataset: {len(test_loader)}")
# Check the size of each batch in the training and testing datasets
for images, labels in train_loader:
    print(f"Batch size in train: {images.size()}")
    break
# Check the size of each batch in the testing dataset
for images, labels in test_loader:
    print(f"Batch size in test: {images.size()}")
    break
# Check the number of classes in the training and testing datasets
print(f"Number of classes in training dataset: {len(train_dataset.classes)}")
print(f"Number of classes in testing dataset: {len(test_dataset.classes)}")

# Vamos a cargar el modelo preentrenado de ResNet18
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
# Load the pre-trained ResNet18 model
model = models.resnet18(weights='DEFAULT')
# Modify the last fully connected layer to match the number of classes in the dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Vamos con la funcion de perdida y el optimizador y epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
# Define the number of epochs
num_epochs = 13

# Initialize lists to store training and testing loss and accuracy
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# Vamos a entrenar el modelo
start_time = time.time()
print("Starting training...")

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # Move data to GPU if available
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = correct / total

    test_loss_list.append(epoch_loss)
    test_acc_list.append(epoch_acc)

    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}") 

end_time = time.time()
print("Training completed.")
print()
print(f"Training completed in {num_epochs} epochs")
print(f"Final training loss: {train_loss_list[-1]:.4f}")
print(f"Final training accuracy: {train_acc_list[-1]:.4f}")
print()
print(f"Training started at: {time.ctime(start_time)}")
print(f"Training ended at: {time.ctime(end_time)}")
print(f"Training time: {end_time - start_time:.2f} seconds")

train_loss_list

train_acc_list

test_loss_list

test_acc_list

# Vamos a guardar el modelo entrenado
model_path = 'resnet18_skin_cancer.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Vamos a graficar loss vs epochs train
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Epochs')
plt.grid()
plt.show()

# Vamos a graficar loss vs epochs test
plt.figure(figsize=(10, 5))
plt.plot(test_loss_list, label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss vs Epochs')
plt.grid()
plt.show()

# Vamos a graficar loss vs epochs train y test
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss vs Epochs')
plt.grid()
plt.show();

# Vamos a guardar la grafica de loss vs epochs train y test
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss vs Epochs')
plt.grid()
plt.savefig('train_test_loss_vs_epochs.png')
plt.close()  # Close the plot to free up memory

# Vamos a graficar accuracy vs epochs train
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Epochs')
plt.grid()
plt.show()

# Vamos a graficar accuracy vs epochs test
plt.figure(figsize=(10, 5))
plt.plot(test_acc_list, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Epochs')
plt.grid()
plt.show()

# Vamos a graficar accuracy vs epochs train y test
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs Epochs')
plt.grid()
plt.show()

# Vamos a guardar la grafica de accuracy vs epochs train y test
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs Epochs')
plt.grid()
plt.savefig('train_test_accuracy_vs_epochs.png')
plt.close()  # Close the plot to free up memory

# Vamos a graficar la matriz de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
# Get the model predictions on the test set
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
# Convert to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=train_dataset.classes, columns=train_dataset.classes)
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()  # Close the plot to free up memory

# Vamos a calcular la accuracy, la precision, el recall y el f1-score
from sklearn.metrics import classification_report
# Calculate the classification report
report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)
# Convert to DataFrame for better visualization
report_df = pd.DataFrame(report).transpose()
# Display the classification report
print(report_df)


# Save the classification report to a CSV file
report_df.to_csv('classification_report.csv', index=True)