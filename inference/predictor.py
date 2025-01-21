
import torch
import numpy as np
from fusion.fusion_utils import early_fusion,late_fusion
from visualization.visual_utils import filter_ground_points 
from detection.common_utils import *
from detection.models import load_data_to_gpu

def boxes_to_corners_3d(boxes3d):
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
     
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d

class FramePredictor:
    def __init__(self, detector):
        self.detector = detector
        self.KITTI_CLASS_COLOR_MAP = {1: 'red', 2: 'green', 3: 'orange'}

    def predict_single_frame(self, model, data_dict):
        with torch.no_grad():        
            boxes_2d, boxes_3d_early = early_fusion(self.detector, 
                                                  data_dict['images'], 
                                                  data_dict['points'], 
                                                  data_dict['calib'])
            
            
            data_dict = self._prepare_data_for_model(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
           
            boxes_3d = self._extract_3d_boxes(pred_dicts[0])
            

            img_shape = self._get_image_shape(data_dict['images'])
            fused_boxes = late_fusion(boxes_2d, boxes_3d, data_dict['calib'][0], img_shape)
            
        return data_dict, pred_dicts, boxes_2d, fused_boxes, boxes_3d_early

    def get_frame_predictions(self, batch_dict, pred_dicts=None, filter_points=True):
        frame_info = {}
        

        frame_info['pc_data'] = self._process_point_cloud(batch_dict['points'], filter_points)
        frame_info['image'] = batch_dict['images'][0].detach().cpu().numpy()
        frame_info['calib'] = batch_dict['calib'][0]
        

        self._add_ground_truth_info(frame_info, batch_dict)
        

        self._add_prediction_info(frame_info, pred_dicts)
        
        return frame_info

    def _prepare_data_for_model(self, data_dict):
        from detection.kitti_dataset import KittiDataset, build_dataloader  # Import here to avoid circular imports
        data_dict = KittiDataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        return data_dict

    

    def _extract_3d_boxes(self, pred_dict):
        boxes_3d = []
        for i, box in enumerate(pred_dict['pred_boxes']):
            boxes_3d.append({
                'corners': boxes_to_corners_3d(box.unsqueeze(0)).squeeze(0).cpu().numpy(),
                'center': box[:3].cpu().numpy(),
                'conf': pred_dict['pred_scores'][i].cpu().numpy(),
                'label': pred_dict['pred_labels'][i].cpu().numpy()
            })
        return boxes_3d

    @staticmethod
    def _get_image_shape(images):
        if torch.is_tensor(images):
            return (images.shape[2], images.shape[3])
        return images.shape[:2]

    def _process_point_cloud(self, points, filter_points):
        pc_data = points[:, 1:].detach().cpu().numpy()
        if filter_points:
            _, pc_data = filter_ground_points(pc_data, heightFromGround=0.25)
        return pc_data

    def _add_ground_truth_info(self, frame_info, batch_dict):
        gt_boxes = batch_dict.get('gt_boxes', None)
        if gt_boxes is not None:
            gt_boxes = gt_boxes[0].detach().cpu().numpy()
            frame_info['gt_corners'] = boxes_to_corners_3d(gt_boxes)

    def _add_prediction_info(self, frame_info, pred_dicts):
        if pred_dicts is not None:
            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            frame_info['pred_corners'] = boxes_to_corners_3d(pred_boxes)
            frame_info['pred_labels'] = pred_dicts[0]['pred_labels'].detach().cpu().numpy()