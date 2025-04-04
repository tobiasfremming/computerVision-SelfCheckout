import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm  # Import tqdm for progress bars

# === Constants ===
DATA_DIR = 'images/NGD_HACK'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 26

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

# === Load and Prepare Data ===
print(f"Loading data from {DATA_DIR}...")
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")

# Calculate sizes for train-validation split (80-20)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

# Create the splits
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                          generator=torch.Generator().manual_seed(SEED))

print(f"Training set: {train_size} images")
print(f"Validation set: {val_size} images")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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

# === Unfreeze for Fine-tuning ===
def unfreeze_model(model):
    print("Unfreezing model for fine-tuning...")
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    # Freeze first 100 layers (approximate conversion from TF code)
    # EfficientNet has different structure in PyTorch, so we freeze the first few blocks
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'features.0.' in name or 'features.1.' in name or 'features.2.' in name:
            param.requires_grad = False
            frozen_count += 1
    
    print(f"Kept {frozen_count} layers frozen for fine-tuning")
    return model

# === Training Function with Progress Bar ===
def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                       leave=False, ncols=100)
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        correct += batch_correct
        
        # Update progress bar
        batch_acc = batch_correct / targets.size(0)
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.4f}"
        })
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# === Validation Function with Progress Bar ===
def validate(model, loader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", 
                       leave=False, ncols=100)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            batch_correct = predicted.eq(targets).sum().item()
            correct += batch_correct
            
            # Update progress bar
            batch_acc = batch_correct / targets.size(0)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}"
            })
    
    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    
    return val_loss, val_acc

# === Training Workflow ===
def train_model(train_loader, val_loader, num_epochs=10):
    # Create model
    model = create_efficientnet_model()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Phase 1: Train with frozen base
    print("\n" + "="*50)
    print("Phase 1: Training with frozen base layers")
    print("="*50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
    
    # Phase 2: Fine-tune with partially unfrozen base
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning with partially unfrozen base layers")
    print("="*50)
    
    model = unfreeze_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
    
    return model

# === Run Training ===
print("\nStarting training process...")
trained_model = train_model(train_loader, val_loader)

# Save the model
print("\nSaving model...")
torch.save(trained_model.state_dict(), "ngd_efficientnet_model.pth")
print("Model saved successfully as 'ngd_efficientnet_model.pth'!")

# Print completion message
print("\n" + "="*50)
print("Training complete!")
print("="*50)