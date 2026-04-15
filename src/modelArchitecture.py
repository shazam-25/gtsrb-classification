import torch
import torch.nn as nn
import torchvision.models as models

# Define device globally for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
  def __init__(self, num_classes=43):
    super(CNNModel, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # Calculate input features for the linear layer dynamically
    # This assumes an input size like 32x32, adjust if different
    self._feature_extractor = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    with torch.no_grad():
      dummy_input = torch.zeros(1, 3, 32, 32) # Assuming 32x32 input images
      n_features = self._feature_extractor(dummy_input).view(-1).shape[0]
    
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


class ResNet50(nn.Module):
  def __init__(self, num_classes=43):
    super(ResNet50, self).__init__()
    # Load pre-trained ResNet50
    self.model = models.resnet50(pretrained=True)

    # Freeze all parameters in the network
    for param in self.model.parameters():
      param.requires_grad = False

    # Replace the final fully connected layer
    # The input features to the fc layer are `self.model.fc.in_features`
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    # Ensure the newly added layer's parameters are trainable
    for param in self.model.fc.parameters():
        param.requires_grad = True

  def forward(self, x):
    return self.model(x)


class MobileNetV2(nn.Module):
  def __init__(self, num_classes=43):
    super(MobileNetV2, self).__init__()
    # Load pre-trained MobileNetV2
    self.model = models.mobilenet_v2(pretrained=True)

    # Freeze all parameters in the network
    for param in self.model.parameters():
      param.requires_grad = False

    # Replace the final classifier layer
    # MobileNetV2's classifier is a Sequential layer, where the last layer is Linear
    in_features = self.model.classifier[-1].in_features
    self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Ensure the newly added layer's parameters are trainable
    for param in self.model.classifier[-1].parameters():
        param.requires_grad = True

  def forward(self, x):
    return self.model(x)

def get_model(model_name, num_classes=43):
    if model_name == "cnn":
        return CNNModel(num_classes).to(device)
    elif model_name == "resnet50":
        return ResNet50(num_classes).to(device)
    elif model_name == "mobilenetv2":
        return MobileNetV2(num_classes).to(device)
    else:
        raise ValueError(f"Model {model_name} not recognized")

