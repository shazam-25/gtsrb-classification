# --------- BEGIN ---------------
# evaluate.py
# -- Evaluation Loop for the model
# -------------------------------

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -- Prerequisites --
# Make directory to save the reports
os.makedirs("reports/figures", exist_ok=True)

# Calculate Accuracy Score
def calculate_accuracy(y_true, y_pred, filename):
  acc_score = accuracy_score(y_true, y_pred)
  with open(filename, "w") as f:
    f.write(f"Accuracy: {acc_score * 100:.3f}%\n")
  print(f"Accuracy: {acc_score * 100:.3f}%")


# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, filename):
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(20, 15))
  sns.heatmap(cm, cmap="YlGnRe")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Traffic Sign (Class ID)")
  plt.ylabel("Actual Traffic Sign (Class ID)")
  plt.show()

  plt.savefig(filename)
  # print(f"Confusion Matrix saved.")
  plt.close()


# Plot Misclassified Samples
def plot_misclassified_samples(misclassified, filename, num=10):
  num = min(num, len(misclassified))
  plt.figure(figsize=(15,5))

  def denormalize(img):
    return img * 0.5 + 0.5

  for i in range(num):
    plt.subplot(2, 5, i+1)
    img, true, pred = misclassified[i]

    img = denormalize(img)
    img = img.transpose(1, 2, 0)

    plt.imshow(img)
    plt.title(f"True: {true}, Pred: {pred}")
    plt.axis("off")

  plt.suptitle(f"Misclassified Samples")
  plt.show()

  plt.savefig(filename)
  plt.close()
  # print("Misclassified samples plot saved.")
 

# Print Classification Report
def print_classification_report(y_true, y_pred, filename):
  # print(classification_report(y_true, y_pred))
  with open(filename, "w") as f:
    f.write(classification_report(y_true, y_pred))


# Find Most Confused Classes
def find_most_confused_classes(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)

  # Ignore diagonal
  np.fill_diagonal(cm, 0)

  # Find top confusion pairs
  pairs = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0]

  print("Most Confused Class Pairs:")
  for i in range(5):
    a, b = pairs[i]
    print(f"Class {a} → Class {b} ({cm[a][b]} times)")


# Main evaluation function
def evaluate_model(model_name):
  print(f"--- Evalaution Report of {model_name} ---")

  print(f"Model Name: {model_name}")

  model = load_trained_model(model_name)
  preds, labels, misclassified = get_predictions(model, test_dataloader)

  # Accuracy Score
  calculate_accuracy(labels, preds, filename=f"reports/{model_name}_accuracy.txt")

  # Confusion Matrix
  plot_confusion_matrix(labels, preds, filename=f"reports/figures/{model_name}_cm.png")

  # Classification Report --> Saved
  print_classification_report(labels, preds, filename=f"reports/{model_name}_classification_report.txt")

  # Misclassified Samples
  plot_misclassified_samples(misclassified, filename=f"reports/figures/{model_name}_misclassified.png", num=10)

  # Most Confused Classes
  find_most_confused_classes(labels, preds)
  print(f"Finished Evaluation for {model_name}.\n")