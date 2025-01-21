# basic imports
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# DL imports
import torch

# plot library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###################################
# FILE CONSTANTS
###################################

HEIGHT_FROM_GROUND = 0.25
CAMERA_HEIGHT = 1.65   # camera height in meters, from Kitti paper
LIDAR_HEIGHT = 1.73    # LIDAR height in meters, from Kitti paper
COORDINATE_AXIS = ['x', 'y', 'z']

INDICES_1 = [5,4,0,1]
INDICES_2 = [6,5,1,2]
INDICES_3 = [7,6,2,3]
INDICES_4 = [4,7,3,0]



PCD_CAM_VIEW = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-0.8, y=0, z=0.25)
    )

PCD_SCENE=dict(
        xaxis=dict(visible=False,range=[0,70]),
        yaxis=dict(visible=False,range=[-40,40]),
        zaxis=dict(visible=False,),
        aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
        aspectratio=dict(x=1, y=1, z=0.05),
)


###################################
# FUNCTION DEFINITIONS  #
###################################

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


####################################################################

def print_data_range(data):
    for i,ax in enumerate(COORDINATE_AXIS):
        print(f"{ax} axis | min = {data[:,i].min()} | max = {data[:,i].max()}")
        
####################################################################

def filter_ground_points(points, heightFromGround):
    # Lidar z-axis is upwards, so lesser distance from 
    # centre is closer to ground. (-ve is closer to ground
    # than +ve values)
    indicesNearGround = points[:,2] < -(LIDAR_HEIGHT - heightFromGround)
    pointsNearGround = points[indicesNearGround, 0:3]
    pointsAboveGround = points[np.logical_not(indicesNearGround), 0:3]
    return pointsNearGround, pointsAboveGround

####################################################################

def lidar_to_rect(calib, lidar_points):
    # lidar_points shape = (N,3)
    return calib.lidar_to_rect(lidar_points)

def rect_to_image(calib, rect_points):
    # rect_points shape = (N,3)
    return calib.rect_to_img(rect_points)[0]

def convert_lidar_boxes_to_image(boxes3d_corners, calib):
    rect_3d_points = lidar_to_rect(calib, boxes3d_corners.reshape(-1,3))
    image_2d_coordinates = rect_to_image(calib, rect_3d_points).reshape(-1,8,2)
    return image_2d_coordinates

####################################################################


def get_scatter3d_plot(x,y,z, mode='lines', marker_size=1, color=None, opacity=1, colorscale=None, **kwargs):
    return go.Scatter3d(x=x, y=y, z=z, mode=mode, hoverinfo='skip',showlegend=False, 
                        marker = dict(size=marker_size, color=color, opacity=opacity, colorscale=colorscale), **kwargs)

def plot_pc_data3d(x,y,z, apply_color_gradient=True, color=None, marker_size=1, colorscale=None, **kwargs):
    if apply_color_gradient:
        color = np.sqrt(x**2 + y **2 + z **2)
    return get_scatter3d_plot(x,y,z, mode='markers', color=color, colorscale=colorscale, marker_size=marker_size, **kwargs)


def plot_box_corners3d(box3d, color,**kwargs):
    return [
        get_scatter3d_plot(box3d[INDICES_1, 0], box3d[INDICES_1, 1], box3d[INDICES_1, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_2, 0], box3d[INDICES_2, 1], box3d[INDICES_2, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_3, 0], box3d[INDICES_3, 1], box3d[INDICES_3, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_4, 0], box3d[INDICES_4, 1], box3d[INDICES_4, 2], color=color, **kwargs),
    ]


def plot_bboxes_3d(boxes3d, box_colors, **kwargs):
    # boxes3d shape = (N,8,3) = bounding box corners in 3d coordinates
    # box_colors = (N) length vector
    boxes3d_objs = []
    for obj_i in range(boxes3d.shape[0]):
        boxes3d_objs.extend(plot_box_corners3d(boxes3d[obj_i], color = box_colors[obj_i], **kwargs))
    return boxes3d_objs


def get_lidar3d_plots(points, pc_kwargs={}, gt_box_corners=None, gt_box_colors=None, 
                      pred_box_corners=None, pred_box_colors=None, **kwargs):
    lidar3d_plots = []
    lidar3d_raw = []
    #  point cloud data
    tmp = plot_pc_data3d(x=points[:,0], y=points[:,1], z=points[:,2], **pc_kwargs)
    lidar3d_plots.append(tmp)
    lidar3d_raw.append(tmp)      
    # gt bounding boxes
    if((gt_box_corners is not None) and (gt_box_colors is not None)):
        lidar3d_plots.extend(plot_bboxes_3d(gt_box_corners, gt_box_colors, **kwargs))  
    # predicted bounding boxes
    if((pred_box_corners is not None) and (pred_box_colors is not None)):
        lidar3d_plots.extend(plot_bboxes_3d(pred_box_corners, pred_box_colors, **kwargs))  
    return lidar3d_plots, lidar3d_raw

#############################################################################


def clip_bboxes_to_image(bboxes_2d, image_width, image_height):
    # bboxes_2d = (N,8,2), image_width and image_height are ints
    bboxes_2d[:,:,0] = np.clip(bboxes_2d[:,:,0], 0, image_width)
    bboxes_2d[:,:,1] = np.clip(bboxes_2d[:,:,1], 0, image_height)
    return bboxes_2d


def get_base64_string(rgb_image):
    pil_img = Image.fromarray((rgb_image * 255).astype(np.uint8)) # PIL image object
    prefix = "data:image/png;base64,"
    with BytesIO() as stream:
        pil_img.save(stream, format="png")
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    return base64_string


def get_scatter_plot(x,y, mode='lines', marker_size=2, color=None, **kwargs):
    return go.Scatter(x=x, y=y, mode=mode, hoverinfo='skip',showlegend=False, 
                        marker = dict(size=marker_size, color=color), **kwargs)

def plot_box_corners2d(box2d, color,**kwargs):
    return [
        get_scatter_plot(box2d[INDICES_1, 0], box2d[INDICES_1, 1], color=color, **kwargs),
        get_scatter_plot(box2d[INDICES_2, 0], box2d[INDICES_2, 1], color=color, **kwargs),
        get_scatter_plot(box2d[INDICES_3, 0], box2d[INDICES_3, 1], color=color, **kwargs),
        get_scatter_plot(box2d[INDICES_4, 0], box2d[INDICES_4, 1], color=color, **kwargs),
    ]


def plot_bboxes_2d(boxes2d, box_colors, **kwargs):
    # boxes2d shape = (N,8,2) = bounding box corners in image coordinates
    # box_colors = (N) length vector
    boxes2d_objs = []
    for obj_i in range(boxes2d.shape[0]):
        boxes2d_objs.extend(plot_box_corners2d(boxes2d[obj_i], color = box_colors[obj_i], **kwargs))
    return boxes2d_objs


def get_image2d_plots(rgb_image, calib, gt_box_corners=None, gt_box_colors=None, 
                      clip_bboxes = False, pred_box_corners=None, pred_box_colors=None):
    image2d_plots = []
    # rgb image
    image2d_plots.append(go.Image(source=get_base64_string(rgb_image), hoverinfo='skip'))  
    
    # gt bounding boxes    
    if((gt_box_corners is not None) and (gt_box_colors is not None)):
        gt_box_corners_image = convert_lidar_boxes_to_image(gt_box_corners, calib )
        if(clip_bboxes):
            gt_box_corners_image = clip_bboxes_to_image(gt_box_corners_image, 
                                image_height=rgb_image.shape[0], image_width=rgb_image.shape[1])
        image2d_plots.extend(plot_bboxes_2d( gt_box_corners_image, gt_box_colors))  
    
    # predicted bounding boxes    
    if((pred_box_corners is not None) and (pred_box_colors is not None)):
        pred_box_corners_image = convert_lidar_boxes_to_image(pred_box_corners, calib)        
        if(clip_bboxes):
            pred_box_corners_image = clip_bboxes_to_image(pred_box_corners_image, 
                                    image_height=rgb_image.shape[0], image_width=rgb_image.shape[1])
        image2d_plots.extend(plot_bboxes_2d(pred_box_corners_image, pred_box_colors))  
    return image2d_plots


