# -------- BEGIN ----------------
# train.py
# -- Training Loop for the model
# -------------------------------

# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# --- Prerequisites ---
# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training function
def train_model(model, train_dataloader, val_dataloader, model_name, epochs=10):
  print(f"\n Training {model_name}...")

  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),  # Only train parameters that require gradients
      lr=0.001
  )

  # Scheduler (reduce LR on plateau)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.3, patience=2
  )

  best_val_loss = float("inf")
  patience = 3
  trigger_time = 0

  for epoch in range(epochs):
    # ------------------
    # Training Phase
    # ------------------
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_dataloader):
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      train_loss += loss.item()

    # ------------------
    # Validation Phase
    # ------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
      for images, labels in tqdm(val_dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

    # Scheduler step
    scheduler.step(val_loss)

    # ---------------------
    # Save Best Model
    # ---------------------
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), f"models/{model_name}.pth")
      print("Model Saved.")

      trigger_time = 0

    else:
      trigger_time += 1
      print(f"No improvement ({trigger_time}/{patience})")

      if trigger_time >= patience:
        print("Early stopping triggered.")
        break

