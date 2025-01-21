import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .visual_utils import get_lidar3d_plots, get_image2d_plots

from .visual_utils import PCD_SCENE, PCD_CAM_VIEW
from fusion.point_processing import *
import os
from inference.predictor import FramePredictor
from detection.common_utils import *



class Visualizer:
    def __init__(self,detector):
        
        self.fig = make_subplots(
            rows=3, cols=1,
            specs=[[{"type": "scatter3d"}], [{}], [{}]], 
            row_heights=[0.4, 0.3, 0.3],
            horizontal_spacing=0.0,
            vertical_spacing=0.0
        )
        
        self.fig.update_layout(
            template="plotly_dark",
            scene=PCD_SCENE,
            scene_camera=PCD_CAM_VIEW,
            height=900,  # Increased for 3 rows
            width=800,
            title=f"KITTI 3D OBJECT DETECTION using - Early Fusion", 
            title_x=0.5, 
            title_y=0.95,
            margin=dict(r=0, b=0, l=0, t=30),
            showlegend=False
        )              
        self.fig.update_xaxes(showticklabels=False, visible=False, row=2, col=1)
        self.fig.update_yaxes(showticklabels=False, visible=False, row=2, col=1)
        self.fig.update_xaxes(showticklabels=False, visible=False, row=3, col=1)
        self.fig.update_yaxes(showticklabels=False, visible=False, row=3, col=1)
        
        self.fig.update_xaxes(range = [-2, 70], row=1, col=1)
        self.fig.update_yaxes(range = [-40,40], row=1, col=1)
        
        # set export image option
        self.fig.to_image(format="png", engine="kaleido")
        
        self.late_fusion_fig = make_subplots(
        rows=3, cols=1,
        specs=[[{"type": "scatter3d"}], [{}], [{}]],  # Added third row spec
        row_heights=[0.4, 0.3, 0.3],  # Adjusted heights for three rows
        horizontal_spacing=0.0,
        vertical_spacing=0.0
    )

        self.late_fusion_fig.update_layout(
            template="plotly_dark",
            scene=PCD_SCENE,
            scene_camera=PCD_CAM_VIEW,
            height=900,  # Increased height to accommodate third row
            width=800,
            title=f"KITTI 3D OBJECT DETECTION using - Late fusion + Lidar",
            title_x=0.5,
            title_y=0.95,
            margin=dict(r=0, b=0, l=0, t=30),
            showlegend=False
        )


        self.late_fusion_fig.update_xaxes(showticklabels=False, visible=False, row=2, col=1)
        self.late_fusion_fig.update_yaxes(showticklabels=False, visible=False, row=2, col=1)
        self.late_fusion_fig.update_xaxes(showticklabels=False, visible=False, row=3, col=1)
        self.late_fusion_fig.update_yaxes(showticklabels=False, visible=False, row=3, col=1)

        self.late_fusion_fig.update_xaxes(range=[-2, 70], row=1, col=1)
        self.late_fusion_fig.update_yaxes(range=[-40, 40], row=1, col=1)


        self.late_fusion_fig.to_image(format="png", engine="kaleido")

        self.pred_color = 'orange'
        self.predictor = FramePredictor(detector)
     
    def clear_figure_data(self):
        self.fig.data = []
        self.late_fusion_fig.data = []
    
    def clip_coordinates(self,x, y, img_shape):
        
        x = np.clip(x, 0, img_shape[1])  # clip to width
        y = np.clip(y, 0, img_shape[0])  # clip to height
        return x, y    
    
    def get_lidar_points_in_3d_boxes(self,pred_corners, point_cloud):
        points_in_boxes = []
        for box_corners in pred_corners:
            # Get points inside this 3D box
            mask = points_in_3d_box(point_cloud[:, :3], box_corners)
            points_in_boxes.append(point_cloud[mask])
        return points_in_boxes    
    
    def add_lidar_plots(self, frame_preds,is_late_fusion=False):
        lidar_3d_plots, lidar3d_raw = get_lidar3d_plots(frame_preds['pc_data'], 
                                   pred_box_corners = frame_preds['pred_corners'], 
                                   pred_box_colors= [self.pred_color] * frame_preds['pred_corners'].shape[0]
                                  )
        for trace in lidar3d_raw:
            self.fig.add_trace(trace, row=1, col=1)
        for trace in lidar_3d_plots:
         if  is_late_fusion:
            self.late_fusion_fig.add_trace(trace, row=1, col=1)

    def add_image_plots(self, frame_preds, boxes_2d,viz_type_early,boxes_3d_early,viz_type_late, viz_type='both'):
        
        img_shape = frame_preds['image'].shape[:2]
    

        image_2d_plots = get_image2d_plots(
            rgb_image=frame_preds['image'],
            calib=frame_preds['calib'],
            clip_bboxes=True
        )

        lidar_projection_plots = get_image2d_plots(
            rgb_image=frame_preds['image'],
            calib=frame_preds['calib'],
            clip_bboxes=True,
            pred_box_corners=frame_preds['pred_corners'],
            pred_box_colors=[self.pred_color] * frame_preds['pred_corners'].shape[0]
        )
        

        points_in_boxes = self.get_lidar_points_in_3d_boxes(frame_preds['pred_corners'], frame_preds['pc_data'])
        for points in points_in_boxes:
            points_2d, _ = frame_preds['calib'].lidar_to_img(points[:, :3])
            lidar_projection_plots.append(
                go.Scatter(x=points_2d[:, 0], y=points_2d[:, 1],
                            mode='markers', marker=dict(color='red', size=2))
            ) 

        if viz_type_early:
            # For 2D detection only row (middle row)
            box_2d_plots = []
            if boxes_2d:
                for box in boxes_2d:
                    x1, y1, x2, y2 = box['box']
                    box_2d_plots.append(
                        go.Scatter(
                            x=[x1, x2, x2, x1, x1],
                            y=[y1, y1, y2, y2, y1],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        )
                    )


            for trace in image_2d_plots + box_2d_plots:
                self.fig.add_trace(trace, row=2, col=1)
                self.late_fusion_fig.add_trace(trace, row=2, col=1) 
                       


            if boxes_3d_early:
                early_fusion_plots = []
                for box in boxes_3d_early:
                    if 'corners_3d' in box and isinstance(box['corners_3d'], np.ndarray):
                        # Project 3D corners to image
                        corners_2d, in_front = frame_preds['calib'].lidar_to_img(box['corners_3d'])
                        
                        if np.all(in_front):
                            # Draw 3D box edges
                            edges = [
                                (0,1), (1,2), (2,3), (3,0),  # Bottom face
                                (4,5), (5,6), (6,7), (7,4),  # Top face
                                (0,4), (1,5), (2,6), (3,7)   # Connect top to bottom
                            ]
                            
                            for start, end in edges:
                               
                                x1, y1 = self.clip_coordinates(
                                    float(corners_2d[start,0]),
                                    float(corners_2d[start,1]),
                                    img_shape 
                                )
                                x2, y2 = self.clip_coordinates(
                                    float(corners_2d[end,0]),
                                    float(corners_2d[end,1]),
                                    img_shape
                                )
                                
                                early_fusion_plots.append(
                                    go.Scatter(
                                        x=[x1, x2],
                                        y=[y1, y2],
                                        mode='lines',
                                        line=dict(color='purple', width=2),
                                        showlegend=False
                                    )
                                )
                            
 
                            if 'points' in box and isinstance(box['points'], np.ndarray):
                                points_2d, _ = frame_preds['calib'].lidar_to_img(box['points'][:, :3])
                                if points_2d is not None and len(points_2d) > 0:
                                    early_fusion_plots.append(
                                        go.Scatter(
                                            x=points_2d[:, 0].astype(float),
                                            y=points_2d[:, 1].astype(float),
                                            mode='markers',
                                            marker=dict(color='green', size=2),
                                            showlegend=False
                                        )
                                    )

                # Add early fusion visualization to bottom row
                for trace in image_2d_plots + early_fusion_plots:
                    self.fig.add_trace(trace, row=3, col=1) 
        

        color_map = {
            'C+L': 'purple',    # Camera + LiDAR fusion
            'C': 'blue',        # Camera only
            'L': 'orange'       # LiDAR only
        }
        
         
        text_color_map = {
            'C+L': 'red',
            'C': 'cyan',
            'L': 'orange'
        }

        if viz_type_late:
            late_fusion_plots = []
            

            # Plot camera-only boxes (2D)
            if 'camera_only_boxes' in frame_preds:
                for box in frame_preds['camera_only_boxes']:
                    x1, y1, x2, y2 = box['box']
                    conf = box.get('conf', 0)
                    
                    # Draw the box
                    late_fusion_plots.append(
                        go.Scatter(
                            x=[x1, x2, x2, x1, x1],
                            y=[y1, y1, y2, y2, y1],
                            mode='lines',
                            line=dict(color=color_map['C'], width=2),
                            showlegend=False
                        )
                    )
                    
                    # Add confidence text
                    text = f'C:{conf:.2f}'
                    center_x = (x1 + x2) / 2
                    
                    late_fusion_plots.append(
                        go.Scatter(
                            x=[center_x],
                            y=[y1 - 10],
                            text=[text],
                            mode='text',
                            textposition='top center',
                            textfont=dict(
                                color=text_color_map['C'],
                                size=13,
                                weight='bold'
                            ),
                            showlegend=False
                        )
                    )


            if 'fused_corners' in frame_preds:
                for i, corners in enumerate(frame_preds['fused_corners']):
                    detection_source = frame_preds.get('detection_sources', ['C+L'] * len(frame_preds['fused_corners']))[i]
                    box_color = color_map.get(detection_source, 'purple')
                    text_color = text_color_map.get(detection_source, 'white')
                    conf = frame_preds.get('confidences', [0] * len(frame_preds['fused_corners']))[i]
                    
                    corners_2d, in_front = frame_preds['calib'].lidar_to_img(corners)
                    if np.all(in_front):
                        # Draw the 3D box
                        edges = [
                            (0,1), (1,2), (2,3), (3,0),
                            (4,5), (5,6), (6,7), (7,4),
                            (0,4), (1,5), (2,6), (3,7)
                        ]
                        
                        for start, end in edges:
                            x1, y1 = self.clip_coordinates(
                                float(corners_2d[start,0]), 
                                float(corners_2d[start,1]), 
                                img_shape
                            )
                            x2, y2 = self.clip_coordinates(
                                float(corners_2d[end,0]), 
                                float(corners_2d[end,1]), 
                                img_shape
                            )
                            
                            if (0 <= x1 <= img_shape[1] and 0 <= y1 <= img_shape[0]) or \
                            (0 <= x2 <= img_shape[1] and 0 <= y2 <= img_shape[0]):
                                late_fusion_plots.append(
                                    go.Scatter(
                                        x=[x1, x2],
                                        y=[y1, y2],
                                        mode='lines',
                                        line=dict(color=box_color, width=2),
                                        showlegend=False
                                    )
                                )
                        
                        # Add confidence text
                        center_x = np.mean(corners_2d[:, 0])
                        center_y = np.min(corners_2d[:, 1])
                        text = f'{detection_source}:{conf:.2f}'
                        
                        late_fusion_plots.append(
                            go.Scatter(
                                x=[center_x],
                                y=[center_y - 10],
                                text=[text],
                                mode='text',
                                textposition='top center',
                                textfont=dict(
                                    color=text_color,
                                    size=13,
                                    weight='bold'
                                ),
                                showlegend=False
                            )
                        )

        

            for trace in image_2d_plots + late_fusion_plots:
                self.late_fusion_fig.add_trace(trace, row=3, col=1)   
     

    def visualize_predictions(self, model, input_data):
        # Get model predictions for input data
        input_data, model_preds, boxes_2d, fused_boxes,boxes_3d_early = self.predictor.predict_single_frame(model, input_data)
        frame_preds = self.predictor.get_frame_predictions(batch_dict=input_data, pred_dicts=model_preds)
        
        if fused_boxes:
            # Separate boxes by their detection source
            lidar_boxes = []
            camera_boxes = []
            fused_source_types = []
            confidences = []
            
            for box in fused_boxes:
                source = box.get('detection_source', 'C+L')
                conf = box.get('conf', 0)
                if source == 'C':
                    camera_boxes.append(box)
                else:  # 'L' or 'C+L'
                    lidar_boxes.append(box)
                    fused_source_types.append(source)
                    confidences.append(conf)
            
            # Stack corners only for boxes that have them (LiDAR and fused boxes)
            if lidar_boxes:
                frame_preds['fused_corners'] = np.stack([box['corners'] for box in lidar_boxes])
                frame_preds['detection_sources'] = fused_source_types
                frame_preds['confidences'] = confidences
            
            # Store camera-only boxes separately
            frame_preds['camera_only_boxes'] = camera_boxes
        

        self.clear_figure_data()
        
        # Add LiDAR plots
        self.add_lidar_plots(frame_preds=frame_preds,is_late_fusion=True)
        
        self.add_image_plots(frame_preds=frame_preds, boxes_2d=boxes_2d,boxes_3d_early=boxes_3d_early, viz_type_early=True,viz_type_late=True)
               
    
    def show_figure(self):
        self.fig.show()
    
    def save_to_png(self, output_path):
        # Ensure the directory exists
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)


        base_path = output_path.rsplit('.', 1)[0] if output_path.endswith('.png') else output_path
        

        early_fusion_path = f"{base_path}_early.png"
        self.fig.write_image(early_fusion_path, format='png')

        if hasattr(self, 'late_fusion_fig'):
            late_fusion_path = f"{base_path}_late.png"
            self.late_fusion_fig.write_image(late_fusion_path, format='png')    
