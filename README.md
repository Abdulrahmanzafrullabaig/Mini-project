<img width="950" alt="image" src="https://github.com/user-attachments/assets/16c68c8a-9a4f-4a2f-a8fd-49a530bff438" />
--

# Signature Verification Model Using ResNet18 and Siamese Network

## Introduction
Signature verification is a critical task in various domains such as banking, legal, and forensic analysis. The goal is to determine whether a given signature is genuine or forged by comparing it with a reference signature. This report outlines the development and evaluation of a signature verification model using a Siamese Network architecture with a ResNet18 backbone. The model leverages deep learning techniques to achieve high accuracy in distinguishing between genuine and forged signatures.

## Components Used

### Backend
- **Python**: The core programming language for implementing the backend logic.
- **Flask**: A lightweight WSGI web application framework used to create the web interface and handle HTTP requests.
- **PyTorch**: An open-source machine learning library used for training and running the deep learning model.
- **TorchVision**: A library containing popular datasets, model architectures, and image transformations for computer vision.
- **PIL (Python Imaging Library)**: Used for image processing tasks.
- **Scikit-learn**: Used for evaluating model performance metrics such as confusion matrix, ROC curve, and AUC.
- **TQDM**: A progress bar library to monitor the training process.
- **Matplotlib and Seaborn**: Libraries used for data visualization.

### Frontend
- **HTML**: Markup language used for structuring the web interface.
- **CSS**: Used for styling the web interface, making it visually appealing and responsive.
- **JavaScript**: Optional for adding dynamic behavior (not heavily used in the current version).

## Siamese Network Architecture

### Overview
A Siamese Network is a type of neural network architecture that contains two or more identical sub-networks. These sub-networks share the same configuration with the same parameters and weights. The primary goal of a Siamese Network is to learn a similarity function that can compare two input vectors and determine their similarity.

### Architecture
In our implementation, the Siamese Network consists of two identical ResNet18 sub-networks. The outputs of these sub-networks are concatenated and passed through fully connected layers to produce a similarity score.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
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
```

### How It Works
1. **Input Images**: Two signature images are passed through the identical ResNet18 sub-networks.
2. **Feature Extraction**: Each sub-network extracts features from the input images.
3. **Concatenation**: The feature vectors from both sub-networks are concatenated.
4. **Similarity Score**: The concatenated feature vector is passed through fully connected layers to produce a similarity score, which indicates whether the signatures are genuine or forged.

## Implementation

### Data Preparation
- **Dataset**: Images of signatures are stored in a specified directory, categorized into subdirectories for each class (e.g., genuine and forged).
- **Transformations**: Images are resized, normalized, and converted to tensors using PyTorch's transforms module.

### Model Training
- **Data Loaders**: Datasets are split into training, validation, and test sets using random_split. DataLoaders are created for efficient batch processing.
- **Training Loop**: The model is trained using an Adam optimizer and Binary Cross-Entropy Loss. A learning rate scheduler and early stopping mechanism are implemented to optimize training.
- **Evaluation**: The model is evaluated using confusion matrix and ROC-AUC metrics. Visualizations are generated using Matplotlib and Seaborn.

```python
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience):
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    test_losses = []
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img1.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validate
        model.eval()
        test_loss = 0.0
        corrects = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * img1.size(0)
                preds = (outputs > 0.5).float()
                corrects += (preds == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        accuracy = corrects / len(test_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Test Loss: {epoch_test_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}')

        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping")
            break

        # Calculate confusion matrix for this epoch
        cm = confusion_matrix(all_labels, all_preds)
        print(f'Confusion Matrix (Epoch {epoch}):\n{cm}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    return train_losses, test_losses, best_accuracy, cm
```

### Inference
- **File Upload**: Users upload reference and verification signatures via the web interface.
- **Prediction**: Uploaded images are pre-processed and passed through the trained model to get prediction probabilities.
- **Result Display**: The results (genuine or forged with percentage confidence) are displayed on the web page.

### Web Interface
- **Form Handling**: A form allows users to upload two images (reference and verification signatures).
- **Responsive Design**: The web interface is designed to be responsive, ensuring usability across various devices.
- **Result Visualization**: Uploaded images and prediction results are displayed to the user.

## Requirements

### Software
- Python 3.7+
- Flask
- PyTorch
- TorchVision
- PIL
- Scikit-learn
- TQDM
- Matplotlib
- Seaborn

### Hardware
- **GPU**: Optional but recommended for training the model to speed up the process.
- **CPU**: Sufficient for running inference.

## Installation
A `requirements.txt` file is provided to install all necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Future Updates

### Enhanced Model Accuracy
- **Data Augmentation**: Implement advanced data augmentation techniques to increase the diversity of training data and improve model robustness.
- **Model Tuning**: Experiment with different model architectures (e.g., deeper networks, ensemble models) and hyperparameter tuning.

### User Interface Improvements
- **Drag-and-Drop**: Add drag-and-drop functionality for file uploads.
- **Progress Indicators**: Implement progress bars or spinners to indicate processing status during uploads and predictions.
- **Enhanced Feedback**: Provide more detailed feedback on prediction confidence and potential reasons for classification.

### Additional Features
- **Multi-class Classification**: Extend the application to handle multiple classes of forgeries (e.g., skilled, unskilled).
- **Signature Comparison History**: Store previous comparisons and allow users to review past results.
- **Authentication and Security**: Implement user authentication to secure access and maintain privacy of uploaded signatures.

### output


## Conclusion
The Signature Verification Model using ResNet18 and Siamese Network provides an effective solution for authenticating signatures. By leveraging deep learning techniques, the model achieves high accuracy in distinguishing between genuine and forged signatures. With continuous improvements and additional features, this application can become a robust tool for various authentication needs in industries such as banking, legal, and forensic analysis.


## Acknowledgements

 I would like to extend my deepest gratitude to Dr. Victor A.I, professor at Maharaja institure of technology Mysore, for his invaluable guidance and support throught the course of this project.
