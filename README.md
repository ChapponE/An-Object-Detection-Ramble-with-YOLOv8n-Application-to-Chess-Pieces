# Chess Piece Detection with YOLOv8n

This project explores adapting the YOLOv8n model for chess piece detection and classification. It includes experiments on model optimization, comparisons with other architectures, and studies of knowledge transfer.

## Project Structure
- **Dataset/** Contains links to chess datasets (chess_data, chess_new) and others (COCO, Vehicle) on Roboflow
- **Notebooks/** Kaggle links to notebooks for experiments and ipynb of the results notebook
- **advance_in_ML_report.pdf** Detailed report of experiments and results
- **Project_MALIA_2024_2025.pdf** Project description

## Experiments
The project includes four main experiments:
- Initial Training - Fine-tuning YOLOv8n on the chess_data dataset (50 epochs)
- Layer Freezing Impact - Performance comparison with 0, 4, 8, 12, 16, and 20 frozen layers
- Architecture Comparison - YOLOv8n vs DETR vs R-CNN
- Knowledge Transfer - Pre-training on different datasets before fine-tuning on chess_data

## Installation and Dependencies
```python
pip install ultralytics
pip install detectron2
pip install roboflow
pip install matplotlib numpy opencv-python
```

## Download Models from Release
You can download the pre-trained models directly from GitHub Releases:

### Method 1: Manual Download
1. Visit the [Releases page](https://github.com/ChapponE/An-Object-Detection-Ramble-with-YOLOv8n-Application-to-Chess-Pieces/releases)
2. Download the desired model file (e.g., `temp_exp1_last.pt`)

### Method 2: Programmatic Download
```python
import os
import requests

def download_model(model_name, output_dir="models"):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Base URL for the release assets
    base_url = "https://github.com/ChapponE/An-Object-Detection-Ramble-with-YOLOv8n-Application-to-Chess-Pieces/releases/download/v1.0.3/"
    
    # Full URL and output path
    url = base_url + model_name
    output_path = os.path.join(output_dir, model_name)
    
    # Download the file
    print(f"Downloading {model_name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Model saved to {output_path}")
    return output_path

# Example: Download the best model
model_path = download_model("temp_exp1_last.pt")
```

## Usage
### Training a Model
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Training
model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    freeze=4
)
```

### Evaluation
```python
results = model.val(data="path/to/data.yaml")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
```

### Inference with Downloaded Model
```python
from ultralytics import YOLO

# Load downloaded model
model = YOLO("models/temp_exp1_last.pt")

# Run inference on an image
results = model("path/to/image.jpg")

# Display results
for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")
    
    # Plot results
    result.show()
    
    # Save results
    result.save("results.jpg")
```

## Authors
```
@misc{ChapponAhrouch2025,
    author = {Chappon, Edouard and Ahrouch, Samira},
    title = {An Object Detection Ramble with YOLOv8n: Application to Chess Pieces},
    year = {2025}
}
```