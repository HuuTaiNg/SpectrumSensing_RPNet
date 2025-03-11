import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])

        # Define the RGB colors for each class
        self.class_colors = {
            (255, 255, 255): 0,     # LTE class
            (127, 127, 127): 1,     # 5G NR class
            (0, 0, 0): 2            # Noise class
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        idx = np.int64(idx)
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None: 
            raise FileNotFoundError(f"Image not found at {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label = cv2.imread(self.label_paths[idx])
        if label is None: 
            raise FileNotFoundError(f"Label not found at {self.label_paths[idx]}")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Map RGB colors to class indices
        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask

# Usage example:
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Resize to desired input size
    transforms.ToTensor()
])

# Creat dataset
train_dataset = SemanticSegmentationDataset(
    image_dir='C:\\Users\\dataset\\train\\input',
    label_dir='C:\\Users\\dataset\\train\\label',
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir='\\dataset\\test\\input',
    label_dir='\\dataset\\test\\label',
    transform=train_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(len(train_dataset))
print(len(val_dataset))
print(len(train_dataloader))
print(len(val_dataloader))
print(len(test_dataloader))

import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.functional as F
import torchvision.models as models

class RPNet_CLRu(nn.Module):
    def __init__(self, n_classes):
        super(myModel, self).__init__()

        # Conv Layer
        self.conv_layer1_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv_layer1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
         
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear")

        self.conv_last1 =  nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2)
        self.batchnorm_last1 = nn.BatchNorm2d(64)
        self.conv_last2 =  nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.batchnorm_last2 = nn.BatchNorm2d(16)
        self.conv_last3 =  nn.Conv2d(16, n_classes, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        
        # =================================================
        x1 = self.conv_layer1_1(x)
        x1 = F.relu(x1)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        x_ = x1
        
        # ===================  Stage 1  ===================
        x2 = self.maxpool_layer(x1)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        # ===================  Stage 2  ===================
        x3_1 = self.maxpool_layer(x2)
        x3_2 = self.maxpool_layer(x1)
        x3_2 = self.maxpool_layer(x3_2)
        x3 = x3_1 + x3_2
        
        x2_ = x2
        x2_1 = self.conv_layer1_2(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.maxpool_layer(x1)
        x2 = x2_1 + x2_2
        
        x1_1 = self.conv_layer1_2(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1 = x1_1 + x1_2
  
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer1_2(x3)
        x3 = F.relu(x3)
        
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer1_2(x3)
        x3 = F.relu(x3)
        
        # ===================  Stage 3  ===================
        x4_1 = self.maxpool_layer(x3)
        x4_2 = self.maxpool_layer(x2)
        x4_2 = self.maxpool_layer(x4_2)
        x4_3 = self.maxpool_layer(x1)
        x4_3 = self.maxpool_layer(x4_3)
        x4_3 = self.maxpool_layer(x4_3)
        x4 = x4_1 + x4_2
        x4 = x4 + x4_3
        
        x3_ = x3
        x3_1 = self.conv_layer1_2(x3)
        x3_1 = F.relu(x3_1)
        x3_2 = self.maxpool_layer(x2)
        x3_3 = self.maxpool_layer(x1)
        x3_3 = self.maxpool_layer(x3_3)
        x3 = x3_1 + x3_2
        x3 = x3 + x3_3
        
        x2_ = x2
        x2_1 = self.conv_layer1_2(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.upsample2(x3_)
        x2_3 = self.maxpool_layer(x1)
        x2 = x2_1 + x2_2
        x2 = x2 + x2_3
        
        x1_1 = self.conv_layer1_2(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1_3 = self.upsample4(x3_)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3

        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer1_2(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer1_2(x4)
        x4 = F.relu(x4)
        
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer1_2(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer1_2(x4)
        x4 = F.relu(x4)
        
        # ===================  Stage 4  ===================
        x4_ = x4
        x4_1 = self.conv_layer1_2(x4)
        x4_1 = F.relu(x4_1)
        x4_2 = self.maxpool_layer(x3)
        x4_3 = self.maxpool_layer(x2)
        x4_3 = self.maxpool_layer(x4_3)
        x4_4 = self.maxpool_layer(x1)
        x4_4 = self.maxpool_layer(x4_4)
        x4_4 = self.maxpool_layer(x4_4)
        x4 = x4_1 + x4_2
        x4 = x4 + x4_3
        x4 = x4 + x4_4
        
        x3_ = x3
        x3_1 = self.conv_layer1_2(x3)
        x3_1 = F.relu(x3_1)
        x3_2 = self.maxpool_layer(x2)
        x3_3 = self.maxpool_layer(x1)
        x3_3 = self.maxpool_layer(x3_3)
        x3_4 = self.upsample2(x4_)
        x3 = x3_1 + x3_2
        x3 = x3 + x3_3
        x3 = x3 + x3_4
        
        x2_ = x2
        x2_1 = self.conv_layer1_2(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.upsample2(x3_)
        x2_3 = self.maxpool_layer(x1)
        x2_4 = self.upsample4(x4_)
        x2 = x2_1 + x2_2
        x2 = x2 + x2_3
        x2 = x2 + x2_4
        
        x1_1 = self.conv_layer1_2(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1_3 = self.upsample4(x3_)
        x1_4 = self.upsample8(x4_)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3
        x1 = x1 + x1_4
 
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer1_2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer1_2(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer1_2(x4)
        x4 = F.relu(x4)
        
        # ===================  Upsampling  ===================
        x1_1 = self.conv_layer1_2(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2)
        x1_3 = self.upsample4(x3)
        x1_4 = self.upsample8(x4)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3
        x1 = x1 + x1_4
        x1 = self.conv_out(x1)
        
        x_ = self.conv(x_)
        x = torch.cat([x_, x1], dim=1)
        x1 = self.batchnorm2(x)
        x1 = F.relu(x1)
        
        x = self.conv_last1(x1)
        x = F.relu(x)
        x = self.conv_last2(x)
        x = F.relu(x)
        x = self.conv_last3(x)
        
        return F.softmax(x)
    
classes = 3
model = RPNet_CLRu(classes)
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassRecall, MulticlassAccuracy, MulticlassPrecision
from torchmetrics import ConfusionMatrix

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Update confusion matrix
        confmat(preds, labels)
        
         # Update metrics
        iou_metric(preds, labels)
        f1_metric(preds, labels)
        accuracy_metric(preds, labels)
        precision_metric(preds, labels)
        
        # Update tqdm description with metrics
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}'
        })
    
    cm = confmat.compute().cpu().numpy()  
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    confmat.reset()

    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()

    epoch_loss = running_loss / len(dataloader.dataset)
    
    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)


    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # Update confusion matrix
            confmat(preds, labels)

            # Update metrics
            iou_metric(preds, labels)
            f1_metric(preds, labels)
            recall_metric(preds, labels)
            accuracy_metric(preds, labels)
            precision_metric(preds, labels)


            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Accuracy': f'{accuracy_metric.compute():.4f}',
                'mIoU': f'{iou_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    cm = confmat.compute().cpu().numpy()  # Convert to numpy for easy usage
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    confmat.reset()

    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()

    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# ===============================  Training model  ===============================
num_epochs = 100
epoch_saved = 0

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs):
    _, epoch_loss_train, iou_score_avg_train, f1_score_avg_train, accuracy_avg_train, precision_avg_train = train_epoch(model, train_dataloader, criterion, optimizer, device, classes)
    _, epoch_loss_val, iou_score_avg_val, f1_score_avg_val, accuracy_avg_val, precision_avg_val = evaluate(model, val_dataloader, criterion, device, classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, mIoU: {iou_score_avg_train:.4f}, Mean F1 Score: {f1_score_avg_train:.4f}, Mean Precision: {precision_avg_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, mIoU: {iou_score_avg_val:.4f}, Mean F1 Score: {f1_score_avg_val:.4f}, Mean Precision: {precision_avg_val:.4f}")
    f = open('training.txt', 'a')
    f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
    f.write(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, mIoU: {iou_score_avg_train:.4f}, Mean F1 Score: {f1_score_avg_train:.4f}, Mean Precision: {precision_avg_train:.4f}\n")
    f.write(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, mIoU: {iou_score_avg_val:.4f}, Mean F1 Score: {f1_score_avg_val:.4f}, Mean Precision: {precision_avg_val:.4f}\n")
    f.close()
    if accuracy_avg_val >= best_val_accuracy:
        epoch_saved = epoch + 1
        best_val_accuracy = accuracy_avg_val
        best_model_state = model.state_dict()

print("===================")
print(f"Best Model at epoch : {epoch_saved}")

