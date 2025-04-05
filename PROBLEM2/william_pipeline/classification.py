# classification_inference.py

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- FROM YOUR MLP CODE ---
class MLPClassifier(nn.Module):
    """A simple 2-layer MLP to classify 2048-d embeddings."""
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=26):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class ClassificationPipeline:
    """
    Combines:
      1) A TF ResNet50-based feature extractor
      2) A PyTorch MLP for classification
    to classify a single bounding-box crop.
    """
    def __init__(self, mlp_weights_path, num_classes=26, device="cpu"):
        self.device = torch.device(device)

        # 1) Load TF ResNet50-based feature extractor
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                    input_shape=(224, 224, 3))
        # Add global average pooling
        self.tf_feature_extractor = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ], name="TF_ResNet50_FeatureExtractor")

        # 2) Build PyTorch MLP & load weights
        self.mlp = MLPClassifier(input_dim=2048, hidden_dim=256, num_classes=num_classes)
        self.mlp.load_state_dict(torch.load(mlp_weights_path, map_location=self.device))
        self.mlp.eval()
        self.mlp.to(self.device)

    def classify_subcrop(self, bgr_image):
        """
        bgr_image: a crop from OpenCV (height, width, 3) in BGR color space.
        Returns: predicted_class (int)
        """
        # a) Convert BGR -> RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # b) Resize to (224, 224)
        rgb_resized = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_AREA)
        # c) Convert to float, expand dims for TF
        rgb_resized = np.float32(rgb_resized)
        # d) Preprocess for ResNet
        rgb_resized = preprocess_input(rgb_resized)   # modifies channels in [−1..1] or something similar
        rgb_expanded = np.expand_dims(rgb_resized, axis=0)  # shape: (1, 224, 224, 3)

        # e) Extract embedding with TF
        embedding = self.tf_feature_extractor(rgb_expanded, training=False)  # shape: (1, 2048)

        # f) Convert embedding -> torch tensor
        emb_torch = torch.tensor(embedding.numpy(), dtype=torch.float32, device=self.device)  # shape: [1, 2048]

        # g) MLP forward pass
        with torch.no_grad():
            logits = self.mlp(emb_torch)  # shape: [1, num_classes]
            _, predicted_class = torch.max(logits, dim=1)

        return int(predicted_class.item())
