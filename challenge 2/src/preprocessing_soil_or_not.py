

"""

Author: Annam.ai IIT Ropar
Team Name: Green Agro
Team Members: 
- Mayank Jain
- Ankit Singh
- Leela Varshitha
Leaderboard Rank: 50

"""












from torchvision import transforms
import torch
from PIL import Image

# Image transformation for feature extraction (224x224, normalized)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Feature extraction function using pretrained ResNet50
def extract_features(image_path, model, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor).cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f" Error with {image_path}: {e}")
        return None
