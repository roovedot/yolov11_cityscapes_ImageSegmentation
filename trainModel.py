from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

# Load a model
model = YOLO("yolo11n-seg.pt")

# Train the model
train_results = model.train(
    data="/media/roovedot/PHILIPS/cityscapes/config.yaml",  # path to dataset YAML
    epochs=1,  # number of training epochs
    batch=0.95, # auto batch size to use 90% of RAM memory
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    amp=True,
    hyp="parameters.yaml",  # path to hyperparameters file
)

