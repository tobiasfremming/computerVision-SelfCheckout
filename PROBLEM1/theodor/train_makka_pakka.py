
import os
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
import glob

DATA_ROOT = os.path.expanduser("./hurtigrutendata_dataset_Kaggle")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "images/test")
DATA_YAML = os.path.join(DATA_ROOT, "dataset.yaml")

# Find and load the latest trained model
model_dirs = glob.glob("runs/detect/yolov11n_hurtigruten*/")
latest_model_dir = max(model_dirs, key=os.path.getctime) if model_dirs else None

model = YOLO("yolo11s.pt")

# Training parameters
ADDITIONAL_EPOCHS = 500       # Number of additional epochs to train
BATCH_SIZE = 64               # Increased batch size to better utilize the A100
IMG_SIZE = 640                # YOLO training image size
SAVE_PERIOD = 20              # Save a checkpoint every 20 epochs to reduce overhead
NUM_WORKERS = 25              # Increase number of dataloader workers

# Determine the next available run number
next_run_number = len(model_dirs) + 1
run_name = f"yolov11n_hurtigruten{next_run_number}"

# Start training with adjustments:
# - workers: more data loader processes
# - half: enables mixed precision training
# - plots: disabled for faster training (set to True if you require visual output)
model.train(
   data=DATA_YAML,
   epochs=ADDITIONAL_EPOCHS,
   imgsz=IMG_SIZE,
   batch=BATCH_SIZE,
   device=0,
   name=run_name,
   plots=True,             # Disable plots to reduce overhead (change to True if needed)
   resume=False,            # Start fresh with the weights from the last run
   save_period=SAVE_PERIOD,
   workers=NUM_WORKERS,     # Increased number of workers for data loading
   half=True                # Enable mixed precision training for faster throughput
)
