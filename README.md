# Occluded Object Detection

## Overview

This project implements the YOLOv1 (You Only Look Once) object detection algorithm to identify and detect objects in images, including those that are occluded. The model divides the input image into a grid, predicts bounding boxes, and classifies the objects within those boxes. This implementation uses PyTorch and includes an end-to-end solution for applying YOLOv1 to images and visualizing the results.

## Features

- **YOLOv1 Model**: Implements the original YOLOv1 architecture with 24 convolutional layers and 2 fully-connected layers.
- **Occluded Object Detection**: Capable of detecting objects even when they are partially obscured.
- **Bounding Box Visualization**: Draws bounding boxes around detected objects with labels and confidence scores.
- **Command Line Interface**: Provides a command-line tool for easy application of the model to images.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- PIL (Pillow)

You can install the required Python packages using pip:

```bash
pip install torch torchvision opencv-python pillow
```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/dhairya99999/OccludedObjectDetection.git
   cd occludedobjectdetection
   ```

2. **Download Pre-trained Weights**

   Download the YOLOv1 model weights and place them in the project directory. Update the `--weights` argument in the script to point to the downloaded weights file.

## Usage

To run the object detection on an image, use the following command:

```bash
python detect.py --weights <path_to_weights> --threshold <confidence_threshold> --split_size <grid_size> --num_boxes <number_of_boxes> --num_classes <number_of_classes> --input <input_image_path> --output <output_image_path>
```

### Arguments

- `--weights`: Path to the YOLOv1 model weights file (e.g., `YOLO_bdd100k.pt`).
- `--threshold`: Confidence threshold for bounding box predictions (default: `0.5`).
- `--split_size`: Size of the grid applied to the image (default: `14`).
- `--num_boxes`: Number of bounding boxes predicted per grid cell (default: `2`).
- `--num_classes`: Number of classes the model is trained to detect (default: `13`).
- `--input`: Path to the input image file (e.g., `input.jpg`).
- `--output`: Path to save the output image with detected objects (e.g., `output.jpg`).

### Example

```bash
python detect.py --weights ./weights/YOLO_bdd100k.pt --threshold 0.5 --split_size 14 --num_boxes 2 --num_classes 13 --input ./images/sample.jpg --output ./images/sample_output.jpg
```

## Code Structure

- `model.py`: Contains the implementation of the YOLOv1 model.
- `YOLO_to_Image.py`: Main script for performing object detection on images.
- `utils.py`: (Optional) Utility functions for image processing and visualization.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or bug fixes.

## Acknowledgements

- YOLOv1 paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- OpenCV: [https://opencv.org/](https://opencv.org/)
