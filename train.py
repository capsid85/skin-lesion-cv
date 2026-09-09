import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import numpy as np

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10 # Change as needed
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "model_weights/best_model.pth"

# 7 classes for HAM10000
NUM_CLASSES = 7
CLASS_MAPPING = {
    'actinic_keratoses': 0,
    'basal_cell_carcinoma': 1,
    'benign_keratosis-like_lesions': 2,
    'dermatofibroma': 3,
    'melanoma': 4,
    'melanocytic_Nevi': 5,
    'vascular_lesions': 6
}

class HAM10000Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert grayscale or RGBA to RGB just in case
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        label_str = item['dx']
        label = CLASS_MAPPING.get(label_str, 5) # default to 'nv' (most common) if missing
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders():
    print("Loading dataset from HuggingFace...")
    # Load dataset. The dataset marmal88/skin_cancer has train, validation splits.
    # Note: Sometimes it might just have 'train'. Let's load train and split it if needed.
    dataset = load_dataset('marmal88/skin_cancer')
    
    # Check if there is a validation/test split, if not, we split the train set.
    if 'validation' in dataset:
        train_data = dataset['train']
        val_data = dataset['validation']
    else:
        # Split 80/20
        split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_data = split['train']
        val_data = split['test']

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HAM10000Dataset(train_data, transform=train_transform)
    val_dataset = HAM10000Dataset(val_data, transform=val_transform)

    # Handling Class Imbalance with Weighted Random Sampler
    print("Calculating class weights for WeightedRandomSampler...")
    train_labels = [CLASS_MAPPING.get(item['dx'], 5) for item in train_data]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def get_model():
    # Load EfficientNet-B0 with pretrained weights
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Modify the classifier head for 7 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model.to(DEVICE)

def train_model():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    train_loader, val_loader = get_data_loaders()
    model = get_model()
    
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model with validation balanced accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()
