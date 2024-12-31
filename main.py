import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torchvision.models as models
import os

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)
        
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        combined = torch.cat((feat1, feat2), 1)
        output = self.fc1(combined)
        return output

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    
    # Update model path to work with Render's file system
    model_path = os.path.join(os.path.dirname(__file__), 'final_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image).unsqueeze(0)

def make_prediction(model, img1, img2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        output = model(img1, img2)
    
    return output.item()