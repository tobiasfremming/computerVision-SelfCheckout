import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm  # Import tqdm for progress bar

# === Constants ===
DATA_DIR = 'images/NGD_HACK_VALIDATION'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 26

torch.backends.cudnn.benchmark = True

# Set random seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Data Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Model Creation ===
def create_efficientnet_model(num_classes=NUM_CLASSES):
    print("Loading pretrained EfficientNet-B3...")
    # Load pretrained EfficientNet-B3
    model = models.efficientnet_b3(pretrained=True)
    
    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=in_features, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=256, out_features=num_classes)
    )
    
    print("Model prepared with frozen base layers and new classifier")
    return model

# Load model architecture
model = create_efficientnet_model()
model = model.to(device)

# Load saved weights
model.load_state_dict(torch.load("ngd_efficientnet_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# === Load and Prepare Data ===
print(f"Loading data from {DATA_DIR}...")
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")

from PIL import Image

# Define the preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Get class names from the training dataset
class_names = dataset.classes  # You already loaded dataset earlier

def predict(image_path):
    image_tensor = process_image(image_path)
    
    with torch.no_grad():  # Turn off gradients for inference
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    predicted_label = class_names[predicted_idx.item()]
    return predicted_label

import os

def evaluate_model_on_folder(root_dir):
    correct = 0
    total = 0

    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir): continue

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            image_path = os.path.join(class_dir, filename)
            predicted = predict(image_path)
            is_correct = (predicted == label)
            if is_correct: correct += 1
            total += 1
            print(f"{filename}: predicted={predicted}, actual={label} -> {'✅' if is_correct else '❌'}")

    print(f"\nOverall accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")

evaluate_model_on_folder(DATA_DIR)