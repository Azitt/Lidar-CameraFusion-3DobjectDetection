from .point_processing import *
import torch

import numpy as np

from scipy.spatial.transform import Rotation

def early_fusion(detector, image, point_cloud, calib):
    # 2D detections
    boxes_2d = detector.detect(image)
    
    
    if point_cloud.shape[0] == 4:  # If points are (4, N)
        point_cloud = point_cloud.T  # Convert to (N, 4)
    
    
    points_3d = point_cloud[:, :3] if point_cloud.shape[1] >= 3 else point_cloud
    
    # Project all LiDAR points to image plane
    points_2d, depths = calib.lidar_to_img(points_3d)
    
    # Filter points in front of camera
    mask = depths >= 0
    valid_points = points_3d[mask]  # Use only 3D coordinates
    valid_points_2d = points_2d[mask]
    valid_depths = depths[mask]

    boxes_3d = []
    for box in boxes_2d:
        try:
            mask = points_in_box_2d(valid_points_2d, box['box'])
            points_in_box = valid_points[mask]
            depths_in_box = valid_depths[mask]
            
            if len(points_in_box) < 10:
                continue
                
           
            x1, y1, x2, y2 = box['box']
            center_2d = np.array([(x1 + x2)/2, (y1 + y2)/2])
            

            depth = np.median(depths_in_box)
            
            # Convert to camera 3D space
            center_cam = calib.img_to_rect(
                u=np.array([center_2d[0]], dtype=np.float32),
                v=np.array([center_2d[1]], dtype=np.float32),
                depth_rect=np.array([depth], dtype=np.float32)
            )
            

            center_lidar = calib.rect_to_lidar(center_cam)
            
            # Adjust height using LiDAR points (z is up in LiDAR frame)
            ground_height = np.percentile(points_in_box[:, 2], 5)
            center_lidar[0, 2] = ground_height + 0.8  # Set center 0.8m above ground
            

            points_local = points_in_box - center_lidar
            
            # Use PCA on xy plane for orientation
            points_xy = points_local[:, :2]
            if len(points_xy) >= 2:
                cov = np.cov(points_xy.T)
                eigenvals, eigenvects = np.linalg.eigh(cov)
                direction = eigenvects[:, np.argmax(eigenvals)]
                angle = np.arctan2(direction[1], direction[0])
                # Normalize angle to align with camera view
                if abs(angle) > np.pi/2:
                    angle = angle - np.sign(angle) * np.pi
            else:
                angle = 0
                
            rotation = Rotation.from_rotvec(angle * np.array([0, 0, 1]))
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Estimate dimensions based on 2D box size and depth
            l = 4.5  # Default length for cars in KITTI
            w = max(1.8, depth * box_width / center_2d[0] * 0.8)  # Scale width based on perspective
            h = max(1.5, depth * box_height / center_2d[1] * 0.6)  # Scale height based on perspective
            
            l = np.clip(l, 3.5, 5.0)    # Length between 3.5m and 5.0m
            w = np.clip(w, 1.5, 2.1)    # Width between 1.5m and 2.1m
            h = np.clip(h, 1.3, 1.8)    # Height between 1.3m and 1.8m
            
            # Point cloud based refinement
            if len(points_in_box) > 10:
                # Project points to object's local frame
                local_points = rotation.inv().apply(points_in_box - center_lidar)
                
                # Refine width and length using point spread
                point_width = np.percentile(np.abs(local_points[:, 1]), 85) * 2.0
                point_length = np.percentile(np.abs(local_points[:, 0]), 85) * 2.0
                
                w = (w + point_width) / 2
                l = (l + point_length) / 2
            
            # Create box with refined dimensions
            corners_3d = create_3d_box(center_lidar[0], l, w, h, rotation)
            
            box_3d = {
                'corners_3d': corners_3d,
                'center_3d': center_lidar[0],
                'dimensions': {'length': float(l), 'width': float(w), 'height': float(h)},
                'orientation': rotation,
                'points': points_in_box
            }
            boxes_3d.append(box_3d)
            
        except Exception as e:
            print(f"Error creating 3D box: {e}")
            continue
    
    return boxes_2d, boxes_3d

def estimate_orientation(points):
    """
    Estimate object orientation from points using PCA
    """   
    points_xy = points[:, :2]
    
    centroid = np.mean(points_xy, axis=0)
    centered_points = points_xy - centroid
    
    # Simple PCA
    cov = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Get major axis direction
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Calculate orientation angle
    angle = np.arctan2(major_axis[1], major_axis[0])
    
    # Return vertical rotation axis and angle
    return np.array([0, 0, 1]), angle

def estimate_dimensions(points, rotation):
    """
    Estimate 3D box dimensions 
    """
    # Default dimensions for a typical car
    default_l, default_w, default_h = 4.5, 1.8, 1.5
    
    if len(points) < 5:
        return default_l, default_w, default_h
        
    # Rotate points to align with orientation
    points_rotated = rotation.apply(points)
    
    # Calculate dimensions from point spread
    l = np.percentile(points_rotated[:, 0], 95) - np.percentile(points_rotated[:, 0], 5)
    w = np.percentile(points_rotated[:, 1], 95) - np.percentile(points_rotated[:, 1], 5)
    h = np.percentile(points_rotated[:, 2], 95) - np.min(points_rotated[:, 2])
    
    l = np.clip(l, 3.5, 5.0)
    w = np.clip(w, 1.6, 2.0)
    h = np.clip(h, 1.4, 1.8)
    
    return l, w, h

def create_3d_box(center, l, w, h, rotation):
    """
    Create 3D box corners with bottom center as reference
    """
    # Create box corners
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_corners = np.array([   0,    0,    0,    0,    h,    h,    h,    h])
    

    corners_3d = np.stack([x_corners, y_corners, z_corners], axis=1)
    
    corners_3d = rotation.apply(corners_3d) + center
    
    return corners_3d

def points_in_box_2d(points_img, box_2d):
    
    x1, y1, x2, y2 = box_2d
    
    mask = (points_img[:, 0] >= x1) & (points_img[:, 0] <= x2) & \
           (points_img[:, 1] >= y1) & (points_img[:, 1] <= y2)
           
    return mask.astype(bool)


def late_fusion(boxes_2d, boxes_3d, calib, img_shape, distance_threshold=50, iou_threshold=0.2):
    fused_boxes = []
    boxes_3d_proj = []
    
    lidar_close_conf_thresh = 0.7
    lidar_far_conf_thresh = 0.5
    camera_conf_thresh = 0.4

    # Project all 3D boxes and store valid ones
    for box_3d in boxes_3d:
        try:
            corners = box_3d['corners'].cpu().numpy() if torch.is_tensor(box_3d['corners']) else box_3d['corners']
            corners_2d, in_front = calib.lidar_to_img(corners)
            
            if np.all(in_front):
                x_min, y_min = np.min(corners_2d, axis=0)
                x_max, y_max = np.max(corners_2d, axis=0)
                center = box_3d['center'].cpu().numpy() if torch.is_tensor(box_3d['center']) else box_3d['center']
                conf = float(box_3d['conf'].cpu()) if torch.is_tensor(box_3d['conf']) else box_3d['conf']
                distance = np.linalg.norm(center[:2])
                
                boxes_3d_proj.append({
                    'box': np.array([x_min, y_min, x_max, y_max]),
                    'distance': distance,
                    'conf': conf,
                    'original': box_3d,
                    'index': len(boxes_3d_proj)
                })
        except Exception as e:
            continue

    # Filter valid camera detections
    valid_boxes_2d = [box for box in boxes_2d if box.get('conf', 0) >= camera_conf_thresh]
    
    matched_3d = set()
    matched_2d = set()
    
    # First attempt: Find high IoU matches for fusion
    for i, box_3d_p in enumerate(boxes_3d_proj):
        best_match = None
        best_iou = iou_threshold
        
        for j, box_2d in enumerate(valid_boxes_2d):
            if j in matched_2d:
                continue
            
            iou = calculate_iou(box_3d_p['box'], box_2d['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = (i, j)
        
        if best_match is not None:
            
            matched_3d.add(best_match[0])
            matched_2d.add(best_match[1])
            
            box_3d = boxes_3d[box_3d_p['index']]
            box_3d_copy = box_3d.copy()
            box_3d_copy['detection_source'] = 'C+L'
            fused_boxes.append(box_3d_copy)
    
    # Process unmatched detections
    for i, box_3d_p in enumerate(boxes_3d_proj):
        if i not in matched_3d:
            conf = box_3d_p['conf']
            distance = box_3d_p['distance']
            
            # Check confidence based on distance
            keep_lidar = False
            if distance > distance_threshold:
                keep_lidar = conf > lidar_far_conf_thresh
            else:
                keep_lidar = conf > lidar_close_conf_thresh
            
            if keep_lidar:
                box_3d_copy = box_3d_p['original'].copy()
                box_3d_copy['detection_source'] = 'L'
                fused_boxes.append(box_3d_copy)

    # Add unmatched camera detections
    for j, box_2d in enumerate(valid_boxes_2d):
        if j not in matched_2d:
            box_2d_copy = box_2d.copy()
            box_2d_copy['detection_source'] = 'C'
            fused_boxes.append(box_2d_copy)

    return fused_boxes

def is_box_visible(box, img_shape):
    """Check if box is within image bounds with some margin"""
    h, w = img_shape[:2]
    margin = 10
    
    if isinstance(box, np.ndarray) and box.ndim == 2:
        # For corner points
        x_visible = np.any((box[:, 0] >= -margin) & (box[:, 0] < w + margin))
        y_visible = np.any((box[:, 1] >= -margin) & (box[:, 1] < h + margin))
    else:
        # For [x_min, y_min, x_max, y_max] format
        x_visible = (box[0] < w + margin) and (box[2] > -margin)
        y_visible = (box[1] < h + margin) and (box[3] > -margin)
    
    return x_visible and y_visible

def calculate_iou(box1, box2):
   """Calculate IoU between two 2D boxes"""
   x1 = max(box1[0], box2[0])
   y1 = max(box1[1], box2[1])
   x2 = min(box1[2], box2[2])
   y2 = min(box1[3], box2[3])
   
   intersection = max(0, x2 - x1) * max(0, y2 - y1)
   area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
   area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
   
   return intersection / (area1 + area2 - intersection)