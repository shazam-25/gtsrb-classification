# 🚦German Traffic Sign Recognition

This project implements a deep learning classification pipeline to identify German traffic signs using the GTSRB dataset.

## 📋Project Overview
* **Task Type:** Multi-class Image Classification
* **Dataset:** [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## 📂Project Folder Structure

```text
gtsrb-sign-classification/
├── data/
│   ├── interim/        # Cropped/resized images before final serialization
│   ├── processed/      # Final tensors or TFRecords ready for training
│       ├── val_dataloader.pt
│       ├── train_dataloader.pt
│       ├── test_dataloader.pt
│   └── raw/            # The original Kaggle download
│       ├── Train/
│       ├── Test/
│       ├── Train.csv
│       └── Test.csv
├── models/             # Saved .pth or .h5 weights and architecture exports
│   ├── cnn.pth
│   ├── 
│   ├── 
├── notebooks/          # EDA and experimentation
│   ├── 01-eda-and-preprocessing.ipynb
│   └── 02-model-prototyping.ipynb
├── src/                # Modular Python scripts
│   ├── __init__.py
│   ├── data_loader.py  # Ingestion logic (Dataset/DataLoader classes)
│   ├── preprocess.py   # Image normalization, CLAHE, and resizing logic
│   ├── train.py        # Main training loop
│   └── evaluate.py     # Metrics (Accuracy, Confusion Matrix, F1-Score, etc.)
├── reports/            # Visualizations, Loss curves, and logs
│   └── figures/
├── requirements.txt    # Library dependencies (torch, torchvision, pandas, etc.)
└── README.md           # Project overview and setup instructions
```

---

## ⚙️ML Operation Flowchart



The workflow follows a modular approach to ensure scalability and reproducibility:

1.  **Data Ingestion:** Raw images are loaded and mapped using `Train.csv`.
2.  **Preprocessing:** Images undergo resizing and normalization (including CLAHE for better contrast) via `preprocess.py`.
3.  **Modeling:** Prototyping is conducted in Jupyter notebooks before being formalized into the `src/` directory.
4.  **Training:** The `train.py` script executes the training loop, saving the best-performing weights to the `models/` folder.
5.  **Evaluation:** Performance is assessed using F1-Scores and Confusion Matrices to identify specific sign misclassifications.

---

## 🚀Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Usage
1.  **Prepare Data:** Place the GTSRB dataset in `data/raw/`.
2.  **Train the Model:**
    ```bash
    python src/train.py
    ```
3.  **Evaluate:**
    ```bash
    python src/evaluate.py
    ```
