import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CarEmbeddingModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Используем предобученную ResNet50 без последнего слоя
        base_model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-1])  # Убираем FC слой
        self.model.eval()
        self.model.to(self.device)
        logger.info("CarEmbeddingModel initialized")

    def extract_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def extract_embedding_from_array(self, image_array):
        # image_array - numpy array (BGR)
        import cv2
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()