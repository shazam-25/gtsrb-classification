# ------------------BEGIN---------------------------
# data-loader.py
# -- Logic for Dataset Creation and Data Augmentation
# -------------------------------------------------------

# Import Libraries
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocess import preprocess_image


# Function to Create Processed Dataset
class GTSRBDataset(Dataset):
  def __init__(self, df, root_dir, transform=None):
    self.df = df
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    img_path = os.path.join(self.root_dir, self.df.iloc[idx]["Path"])
    label = self.df.iloc[idx]["ClassId"]

    image = cv2.imread(img_path)

    if image is None:
      raise ValueError(f"Image not found.")

    # Apply Image Pre-processing
    image = preprocess_image(img_path)
    # Apply Transformation only to Train Dataset
    if self.transform:
      image = self.transform(image)

    return image, label
  
def get_transforms():
    # Train Transform (with augmentation)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),                                    # Convert to PIL Image for augmentation
        transforms.RandomRotation(degrees=10),                      # Random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),       # Random brightness and contrast
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2)),    # Random crop and resize
        transforms.ToTensor(),                                      # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))                        # Normalize to [-1, 1]
    ])

    # Validation/Test Transform (NO augmentation)
    val_test_transforms = transforms.Compose([
        transforms.ToPILImage(),                # Convert to PIL Image
        transforms.ToTensor(),                  # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1]
    ])

    return train_transforms, val_test_transforms


# -- END OF FILE --