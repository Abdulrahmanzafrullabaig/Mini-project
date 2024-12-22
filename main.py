import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
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

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_signatures(image_path1, image_path2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    # Load model
    model = load_model('best_model.pth')
    
    # Make prediction
    with torch.no_grad():
        output = model(img1, img2)
        prediction = output.item() > 0.5
        confidence = output.item() if prediction else 1 - output.item()
    
    return prediction, confidence * 100