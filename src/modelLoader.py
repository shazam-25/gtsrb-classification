import torch
import numpy as np
from tqdm import tqdm
from src.modelArchitecture import get_model, device # Import device and get_model

# Load Trained Model
def load_model(model_name):
  # Get model architecture and load the trained weights
  model = get_model(model_name)
  model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location=device))
  model.eval()

  # Move the model to the correct device
  model = model.to(device)

  return model

# Get Predictions
def get_predictions(model, dataloader):
  all_preds = []
  all_labels = []
  misclassified = []

  with torch.no_grad():
    for images, labels in tqdm(dataloader):
      images, labels = images.to(device), labels.to(device) # Use the imported device

      outputs = model(images)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

      # Store misclassified samples
      for i in range(len(preds)):
        if preds[i] != labels[i]:
          misclassified.append((images[i].cpu().numpy(), labels[i].item(), preds[i].item()))

  return np.array(all_preds), np.array(all_labels), misclassified
