import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from src.modelLoader import load_model, get_predictions
from src.modelArchitecture import get_model, device # Import device too

# -------------
# Accuracy
# -------------
def calculate_accuracy(y_true, y_pred, filename):
  acc_score = accuracy_score(y_true, y_pred)
  with open(filename, "w") as f:
    f.write(f"Accuracy: {acc_score * 100:.3f}%\n")
  print(f"Accuracy: {acc_score}")

# -----------------
# Confusion Matrix
# -----------------
def plot_confusion_matrix(y_true, y_pred, filename):
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 8))
  sns.heatmap(cm, cmap="YlGnBu")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Traffic Sign (Class ID)")
  plt.ylabel("Actual Traffic Sign (Class ID)")
  plt.show()
  plt.savefig(filename)
  plt.close()

# ----------------------------
# Plot Misclassified Samples
# ----------------------------
def plot_misclassified_samples(misclassified, filename, num=10):
  num = min(num, len(misclassified))
  plt.figure(figsize=(15,5))

  def denormalize(img):
    return img * 0.5 + 0.5

  for i in range(num):
    plt.subplot(2, 5, i+1)
    img, true, pred = misclassified[i]

    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    img = denormalize(img)
    img = np.transpose(img, (1, 2, 0))

    plt.imshow(img)
    plt.title(f"True: {true}, Pred: {pred}")
    plt.axis("off")

  plt.suptitle(f"Misclassified Samples")
  plt.show()
  plt.savefig(filename)
  plt.close()

# ----------------------
# Classification Report
# ----------------------
def print_classification_report(y_true, y_pred, filename):
  with open(filename, "w") as f:
    f.write(classification_report(y_true, y_pred))

# ---------------------------
# Find Most Confused Classes
# ---------------------------
def find_most_confused_classes(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  np.fill_diagonal(cm, 0)
  sorted_indices = np.argsort(cm.ravel())[::-1]
  rows, cols = np.unravel_index(sorted_indices, cm.shape)
  pairs = list(zip(rows, cols))

  print("Most Confused Class Pairs:")
  for i in range(min(5, len(pairs))):
    a, b = pairs[i]
    if cm[a][b] > 0:
      print(f"Class {a} -> Class {b} ({cm[a][b]} times)")


# Run Evaluation Function
def evaluate_model(model_name, test_dataloader):
  os.makedirs("reports/figures", exist_ok=True)
  print(f"--- Evaluation Report of {model_name} ---")
  print(f"Model Name: {model_name}")

  model = load_model(model_name)
  preds, labels, misclassified = get_predictions(model, test_dataloader)

  calculate_accuracy(labels, preds, filename=f"reports/{model_name}_accuracy.txt")
  plot_confusion_matrix(labels, preds, filename=f"reports/figures/{model_name}_cm.png")
  print_classification_report(labels, preds, filename=f"reports/{model_name}_classification_report.txt")
  plot_misclassified_samples(misclassified, filename=f"reports/figures/{model_name}_misclassified.png", num=10)
  find_most_confused_classes(labels, preds)
  print(f"Finished Evaluation for {model_name}.\n")
