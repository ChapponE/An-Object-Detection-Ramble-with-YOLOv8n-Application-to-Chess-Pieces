# Chess Piece Detection with YOLOv8n

This project explores adapting the YOLOv8n model for chess piece detection and classification. It includes experiments on model optimization, comparisons with other architectures, and studies of knowledge transfer.

## Project Structure
- **Dataset/** Contains links to chess datasets (chess_data, chess_new) and others (COCO, Vehicle) on Roboflow
- **Notebooks/** Kaggle links to notebooks for experiments and ipynb of the results notebook
- **Weights/** Trained model weights
- **advance_in_ML_report.pdf** Detailed report of experiments and results
- **Project_MALIA_2024_2025.pdf** Project description

## Experiments
The project includes four main experiments:
- Initial Training - Fine-tuning YOLOv8n on the chess_data dataset (50 epochs)
- Layer Freezing Impact - Performance comparison with 0, 4, 8, 12, 16, and 20 frozen layers
- Architecture Comparison - YOLOv8n vs DETR vs R-CNN
- Knowledge Transfer - Pre-training on different datasets before fine-tuning on chess_data

## Installation and Dependencies
```pip install ultralytics
pip install detectron2
pip install roboflow
pip install matplotlib numpy opencv-python```

## Usage
### Training a Model
```from ultralytics import YOLO
model = YOLO("yolov8n.pt")
Training
model.train(
data="path/to/data.yaml",
epochs=50,
imgsz=640,
batch=16,
freeze=4
)```

### Evaluation
```results = model.val(data="path/to/data.yaml")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
results = model("path/to/image.jpg")```

## Authors
```@misc{ChapponAhrouch2025,
author = {Chappon, Edouard and Ahrouch, Samira},
title = {An Object Detection Ramble with YOLOv8n: Application to Chess Pieces},
year = {2025}
}```
