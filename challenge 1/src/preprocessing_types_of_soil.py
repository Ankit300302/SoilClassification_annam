"""

Author: Annam.ai IIT Ropar
Team Name: Green Agro
Team Members: 
- Mayank Jain
- Ankit Singh
- Leela Varshitha
Leaderboard Rank: 118

"""














from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

# Transformations for training and testing
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset for soil classification
class SoilDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels = labels_df
        self.transform = transform
        self.classes = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        label = self.class_to_idx[self.labels.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label
