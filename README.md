This repository compares two Lidar-camera fusion strategies - early fusion and late fusion - for 3D object detection - through a code implementation using the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), the [PV-RCNN++](https://arxiv.org/pdf/2102.00463) (lidar 3d object detection model) and [Yolov8](https://docs.ultralytics.com/models/yolov8/) (camera 2d object detection model) to evaluate each approach.

## Inference Results: Early vs Late Fusion

<div align="center">
  <strong style="display: inline-block; margin: 0 20px;">Early Fusion – 3D Object Detection Results</strong> 
</div>

<div align="center"> 
  <img src="./resources/early_fusion.gif" alt="Bottom Left Video" width="600"/> 
</div>

<p align="center">
  <strong style="display: inline-block; margin: 0 20px;">Late Fusion – 3D Object Detection Results</strong>
</p>

<div align="center"> 
  <img src="./resources/late_fusion.gif" alt="Bottom Right Video" width="600"/> 
</div>

## Understanding Fusion Approaches
**Early Fusion**
- Early fusion combines raw or low-level features from different sensors at the beginning of the detection pipeline. In our implementation, we:
- Start with 2D object detection (Yolov8) from camera images
- Project LiDAR points into the image plane
- Associate LiDAR points with 2D detections
- Use these points to estimate 3D properties like position, dimensions, and orientation

**Late Fusion**
- Late fusion keeps sensor processing streams separate and combines their high-level outputs. Our implementation:
- Processes camera images to get 2D detections (Yolov8)
- Uses PV-RCNN++ to get 3D detections from LiDAR
- Combines detections using IoU matching and confidence scores




For detailed analysis, check our Medium article.
## Requirements

Python 3.8+
PyTorch
CUDA support
[KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
[Yolov8 pretrained weights](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) I didn’t notice any accuracy difference between YOLOv8 and YOLOv11. I used v8, but feel free to use v11 if it’s easier to find documentation or pretrained weights for it.
[PV-RCNN++](https://github.com/open-mmlab/OpenPCDet/tree/master?tab=readme-ov-file)

## Installation
git clone [repo-url]
cd [repo-name]
pip install -r requirements.txt

## Demo
python main.py

## hello world
