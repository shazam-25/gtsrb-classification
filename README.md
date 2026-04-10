## German Traffic Sign Recongition 
<b>Task Type:</b> Classification
<b>Dataset Link:</b> https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

### Project Folder Structure
gtsrb-sign-classification/
├── data/
│   ├── external/       # Any extra sign datasets (optional)
│   ├── interim/        # Cropped/resized images before final serialization
│   ├── processed/      # Final tensors or TFRecords ready for training
│   └── raw/            # The original Kaggle download (zipped/unzipped)
│       ├── Train/
│       ├── Test/
│       └── Train.csv
├── models/             # Saved .pth or .h5 weights and architecture exports
├── notebooks/          # EDA and experimentation
│   ├── 01-eda-and-preprocessing.ipynb
│   └── 02-model-prototyping.ipynb
├── src/                # Modular Python scripts
│   ├── __init__.py
│   ├── data_loader.py  # Ingestion logic (Dataset/DataLoader classes)
│   ├── preprocess.py   # Image normalization, CLAHE, and resizing logic
│   ├── train.py        # Main training loop
│   └── evaluate.py     # Metrics (Confusion Matrix, F1-Score)
├── reports/            # Visualizations, Loss curves, and logs
│   └── figures/
├── requirements.txt    # Library dependencies (torch, torchvision, pandas, etc.)
└── README.md           # Project overview and setup instructions

### ML Operation Flowchart
