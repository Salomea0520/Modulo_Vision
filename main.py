from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot
import numpy

# 1. Defino los transforms para Train y Test
if __name__ == '__main__':
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),            # Redimensiona a lo que espera ResNet18
        transforms.RandomHorizontalFlip(),        # Reflexión aleatoria
        transforms.RandomResizedCrop(224),        # Recorte aleatorio
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalización estándar ImageNet
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
    # 2. Cargo datasets
train_dir = 'C:/Users/merce/Downloads/skin_cancer_dataset/train-20240324T151905Z-001/train'
test_dir = 'C:/Users/merce/Downloads/skin_cancer_dataset/test-20240324T151902Z-001/test'
train_dataset = datasets.ImageFolder(train_dir, transforms_train)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

# Cargo el modelo ResNet18 preentrenado
model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
print(num_features)  # Mostramos el número de features de entrada a la última capa

# Reemplazo la última capa por una capa con 2 salidas (benigno/maligno)
model.fc = torch.nn.Linear(512, 2)
model = model.to('cpu') #Acuordarse de no usar cpu que sino se peta

# Definimos la función de pérdida
criterion = torch.nn.CrossEntropyLoss()

# Optimizador antes era SGD (con tasa de aprendizaje baja y momentum), ahora Adam
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001) #Esto se ha cambiado respecto al ppt.

#

