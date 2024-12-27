import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torchvision.models as models
import os

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Initialize with default weights to avoid pretrained model download issues
        self.cnn = models.resnet50(weights=None)
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
    print("Starting model loading process...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize model
        model = SiameseNetwork()
        print("Model architecture initialized")
        
        # Get absolute path to the model file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'final_model.pth')
        print(f"Looking for model file at: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load state dict with detailed error handling
        try:
            state_dict = torch.load(model_path, map_location=device)
            print("Model state dict loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading state dict: {str(e)}")
            
        # Load state dict into model
        try:
            model.load_state_dict(state_dict)
            print("State dict loaded into model successfully")
        except Exception as e:
            raise Exception(f"Error loading state dict into model: {str(e)}")
            
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        print("Model successfully moved to device and set to eval mode")
        
        return model
        
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise

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
    device = next(model.parameters()).device  # Get the device model is on
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        output = model(img1, img2)
    
    return output.item()
