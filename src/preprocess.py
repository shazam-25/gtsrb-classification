# ------- BEGIN --------------------
# preprocess.py
# -- Logic for Image Preprocessing
# ----------------------------------

# Import Libraries
import cv2


# Function to Resize Images
def resize_image(img, size=(32,32)):
  return cv2.resize(img, size)


# Function for Color Conversion (BGR --> RGB)
def convert_color(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Function for Contrast Enhancement
def apply_clahe(img):
  # Convert to YUV
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  # Apply CLAHE on Y channel
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img[:, :, 0] = clahe.apply(img[:, :, 0])
  # Convert back to RGB
  img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

  return img


# Function to Normalize Pixel values
def normalize_pixel_values(img):
  return img / 255.0


# Main Preprocessing Function
def preprocess_image(path):

  img = cv2.imread(path)
  img = resize_image(img)
  img = convert_color(img)
  img = apply_clahe(img)
  img = normalize_pixel_values(img)

  return img

# -- END OF FILE --