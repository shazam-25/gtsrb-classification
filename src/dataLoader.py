# ------------------BEGIN---------------------------
# data-loader.py
# -- Logic for Dataset Creation and Data Augmentation
# -------------------------------------------------------

# Import Libraries
import os
import cv2
import torch
from torch.utils.data import Dataset
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
   
# -- END OF FILE --