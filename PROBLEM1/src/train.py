import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input

from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ===========================
#       HYPERPARAMETERS
# ===========================
TRAIN_DIR = 'images/NGD_HACK_TRAIN_CROPPED'
TEST_DIR = 'images/NGD_HACK_VALIDATION_CROPPED'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123
NUM_CLASSES = 26  # Number of classes in your dataset, e.g. A-Z

# ===========================
#     DEVICE & SEED SETUP
# ===========================
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===========================
#   EMBEDDING DATASET CLASS
# ===========================
class EmbeddingDataset(Dataset):
    """
    Holds (embedding, label) pairs in memory for PyTorch.
    """
    def __init__(self, embeddings, labels):
        super().__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        y = self.labels[idx]
        # Convert to Torch tensors
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return x_t, y_t

# ===========================
#   MLP CLASSIFIER IN PYTORCH
# ===========================
class MLPClassifier(nn.Module):
    """A simple 2-layer MLP to classify embeddings."""
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
    
    
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


# ===========================
#   TRAINING LOOP (1 EPOCH)
# ===========================
def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                        leave=False, ncols=100)
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        correct += batch_correct
        
        batch_acc = batch_correct / targets.size(0)
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{batch_acc:.4f}"
        })
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# ===========================
#   VALIDATION LOOP (1 EPOCH)
# ===========================
def validate(model, loader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
            
            batch_acc = batch_correct / targets.size(0)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}"
            })
    
    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    
    return val_loss, val_acc

# ===========================
#   MAIN TRAINING FUNCTION
# ===========================
def train_model(train_loader, val_loader=None, num_epochs=EPOCHS):
    # Create MLP for classifying embeddings (2048 -> 256 -> NUM_CLASSES)
    model = MLPClassifier(input_dim=2048, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
    
    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

    return model

# ===========================
#           MAIN
# ===========================
if __name__ == "__main__":
    # ---------------------------------------------
    # 1) CREATE TF IMAGE GENERATORS (NO LOOP)
    # ---------------------------------------------
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Turn off shuffle so labels remain consistent
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    num_train_samples = train_generator.samples
    num_test_samples = test_generator.samples
    num_classes = train_generator.num_classes

    print(f"\nTrain samples: {num_train_samples}, "
          f"Test samples: {num_test_samples}, Classes: {num_classes}")

    # ---------------------------------------------
    # 2) EXTRACT EMBEDDINGS (ONCE)
    # ---------------------------------------------
    print("\nExtracting training embeddings...")
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
    pool_layer = GlobalAveragePooling2D()
    feature_extractor = Sequential([base_model, pool_layer], name="FeatureExtractor")

    X_train_embed_list = []
    y_train_list = []

    steps_train = len(train_generator)  # Steps in one epoch for training data
    for i in range(steps_train):
        X_batch, y_batch = train_generator[i]  # (batch_size, 224, 224, 3)
        embeddings = feature_extractor.predict(X_batch)  # shape: (batch_size, 2048)
        X_train_embed_list.append(embeddings)
        y_train_list.append(y_batch)

    X_train_embed = np.concatenate(X_train_embed_list, axis=0)  # (num_train_samples, 2048)
    y_train = np.concatenate(y_train_list, axis=0)             # (num_train_samples, num_classes)
    y_train_int = np.argmax(y_train, axis=1)                   # integer labels

    print("X_train_embed shape:", X_train_embed.shape)
    print("y_train_int shape:", y_train_int.shape)

    # ---------------------------------------------
    # 3) APPLY SMOTE (ONCE)
    # ---------------------------------------------
    print("\nApplying SMOTE on the training embeddings...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_embed, y_train_int)

    print("After SMOTE:")
    print("X_train_sm shape:", X_train_sm.shape)
    print("y_train_sm shape:", y_train_sm.shape)

    # ---------------------------------------------
    # 4) PREP TEST EMBEDDINGS (ONCE) - OPTIONAL
    # ---------------------------------------------
    print("\nExtracting test embeddings...")
    X_test_embed_list = []
    y_test_list = []

    steps_test = len(test_generator)
    for i in range(steps_test):
        X_batch, y_batch = test_generator[i]
        embeddings = feature_extractor.predict(X_batch)
        X_test_embed_list.append(embeddings)
        y_test_list.append(y_batch)

    X_test_embed = np.concatenate(X_test_embed_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    y_test_int = np.argmax(y_test, axis=1)

    print("X_test_embed shape:", X_test_embed.shape)
    print("y_test_int shape:", y_test_int.shape)

    # ---------------------------------------------
    # 5) CREATE PYTORCH DATASETS & DATALOADERS
    # ---------------------------------------------
    train_dataset = EmbeddingDataset(X_train_sm, y_train_sm)
    test_dataset  = EmbeddingDataset(X_test_embed, y_test_int)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    # ---------------------------------------------
    # 6) TRAIN PYTORCH MODEL (MULTIPLE EPOCHS)
    # ---------------------------------------------
    print("\nStarting PyTorch MLP training on extracted (SMOTE'd) embeddings...\n")
    trained_model = train_model(train_loader, val_loader=test_loader, num_epochs=EPOCHS)

    # ---------------------------------------------
    # 7) SAVE THE MODEL
    # ---------------------------------------------
    print("\nSaving PyTorch MLP model to 'embedding_based_classifier.pth'...")
    torch.save(trained_model.state_dict(), "embedding_based_classifier.pth")
    print("Model saved successfully!")
