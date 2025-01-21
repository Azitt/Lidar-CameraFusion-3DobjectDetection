import numpy as np

def points_in_box_2d(points_img_tuple, box):
    points_img = points_img_tuple
    x1, y1, x2, y2 = box['box'].astype(float)
    mask = (points_img[:, 0] >= x1) & (points_img[:, 0] <= x2) & \
           (points_img[:, 1] >= y1) & (points_img[:, 1] <= y2)
    return mask

def crop_image(image, box):
    x1, y1, x2, y2 = box['box'].astype(int)
    return image[y1:y2, x1:x2]

def points_in_3d_box(points, box_corners):
   
   box_center = box_corners.mean(axis=0)
   box_dims = box_corners.max(axis=0) - box_corners.min(axis=0)
   
   # Get points relative to box center
   points_centered = points - box_center
   
   # Check if points are within box dimensions
   mask = (abs(points_centered) <= box_dims/2).all(axis=1)
   return mask