import os
import cv2
from tqdm import tqdm
import torch
from visualization.visualizer import Visualizer


import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict

# plot library imports
import plotly.graph_objects as go

### detectors and fusion#####################################
from detection.yolo_detector import YOLODetector
from fusion.fusion_utils import *
from fusion.point_processing import * 
from detection.models import build_network
from visualization.visual_utils import get_lidar3d_plots
from visualization.visual_utils import PCD_SCENE, PCD_CAM_VIEW
from detection.kitti_dataset import KittiDataset, build_dataloader

KITTI_CLASS_COLOR_MAP = {1 : 'red', 2 : 'green', 3 : 'orange'}
 
    
print("Visible CUDA devices:")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    ## **Images**####################################
    img_file = "./kitti_3d_dataset/training/image_2/000149.png"
    img_data = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    plt.imshow(img_data)
    plt.show()

    ##**Point Clouds**###############################
    points_path = "./kitti_3d_dataset/training/velodyne/000149.bin"
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

    lidar_3d_plots,lidar3d_raw = get_lidar3d_plots(points)
    layout = dict(template="seaborn", scene=PCD_SCENE, scene_camera = PCD_CAM_VIEW, title="POINT CLOUD VISUALIZATION")
    fig = go.Figure(data=lidar3d_raw, layout=layout)
    fig.write_image("first_plot_ff.png")

    #### Dataset & Dataloader ###############################
    DATASET_ROOT_DIR = './kitti_3d_dataset/'

    # read config file
    with open('./detection/pvrcnn_config.json', 'r') as f:
        config = json.load(f)
    config = EasyDict(config)

    # train split
    train_dataset = KittiDataset(dataset_cfg=config['DATA_CONFIG'], root_path=DATASET_ROOT_DIR, split='train', training=True)
    train_loader = build_dataloader(train_dataset, batch_size=4, shuffle=True, workers=2)

    # validation split
    val_dataset = KittiDataset(dataset_cfg=config['DATA_CONFIG'], root_path=DATASET_ROOT_DIR, split='val', training=False)
    val_loader = build_dataloader(val_dataset, batch_size=1, shuffle=True, workers=2)

 
    print(len(train_dataset))
    idx = 699

    # dictionary containing point cloud, rgb, ground truth bounding boxes
    sample_data = train_dataset[idx]
    print('Sample data keys : ', sample_data.keys())

    #### shapes of different components in data sample
    pc_data = sample_data['points']
    rgb_image = sample_data['images']
    calib = sample_data['calib']
    print(f"points shape = {pc_data.shape}")
    print(f"images shape = {rgb_image.shape}")


    ### Early Visualization#####################
    lidar_3d_plots,lidar3d_raw = get_lidar3d_plots(pc_data)
    layout = dict(template="plotly_dark", scene=PCD_SCENE, scene_camera = PCD_CAM_VIEW, title="POINT CLOUD VISUALIZATION")
    fig = go.Figure(data=lidar_3d_plots, layout=layout)
    # fig.show()
    fig.write_image("Early_Visualization.png")

    #### ARCHITECTURE#######################


    detector = YOLODetector()

    model = build_network(model_cfg= config.MODEL, num_class=len(config.CLASS_NAMES), dataset=train_dataset)

    ##### WEIGHTS ############################
    model.cuda()
    model.load_params_from_file(filename='./lidar-camera-models/pv_rcnn_8369.pth', device=device)
    model.eval()

    ### INFERENCE#############################

    test_dataset = KittiDataset(dataset_cfg=config['DATA_CONFIG'], root_path=DATASET_ROOT_DIR, split='test', training=False)
 
    # Set up
    visualizer = Visualizer(detector)

    print("test_dataset[0]", test_dataset[0].keys())
    visualizer.visualize_predictions(model, test_dataset[0])

    # Save separate images
    visualizer.save_to_png("tmp/visualization_fusion.png")

if __name__ == "__main__":
    main()