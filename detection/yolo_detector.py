from ultralytics import YOLO
import numpy as np
import cv2

class YOLODetector:
    def __init__(self, model_path='./lidar-camera-models/yolov8x.pt', confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect(self, image):
 
        img = (image * 255).astype(np.uint8)    
        img = cv2.resize(img, (640, 640))
        results = self.model(img, conf=self.confidence)
        
        scale_x = image.shape[1] / 640
        scale_y = image.shape[0] / 640
        
        return self._process_results(results, scale_x, scale_y)    

    def _process_results(self, results, scale_x=1.0, scale_y=1.0):
        boxes = []
        relevant_classes = {0: 'person', 1: 'bicycle', 2: 'car',7: 'truck',9: 'traffic light',11: 'stop sign'}  # KITTI relevant COCO classes
        
        for r in results:
            boxes_data = r.boxes.data.cpu().numpy()
            for box_data in boxes_data:
                x1, y1, x2, y2, conf, cls = box_data
                if int(cls) in relevant_classes:
                    boxes.append({
                        'box': np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]),
                        'conf': conf,
                        'cls': int(cls),
                        'class_name': relevant_classes[int(cls)]
                    })
        return boxes
    
    
    