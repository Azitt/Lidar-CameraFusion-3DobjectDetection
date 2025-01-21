import torch
import numpy as np

class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
            y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
            x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict
    anchor_generator_cfg = [
        EasyDict({
            "class_name": "Car",
            "anchor_sizes": [[3.9, 1.6, 1.56] ],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-1.78],
            "align_center": False,
            "feature_map_stride": 8,
            "matched_threshold": 0.6,
            "unmatched_threshold": 0.45            
        }),
        EasyDict({
            "class_name": "Pedestrian",
            "anchor_sizes": [[0.8, 0.6, 1.73] ],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            "align_center": False,
            "feature_map_stride": 8,
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35            
        }),
        EasyDict({
            "class_name": "Cyclist",
            "anchor_sizes": [[1.76, 0.6, 1.73] ],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            "align_center": False,
            "feature_map_stride": 8,
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35            
        }),                
    ]

    point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1])
    VOXEL_SIZE = np.array([0.05, 0.05, 0.1])
    grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / VOXEL_SIZE
    feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]

    A = AnchorGenerator(
        anchor_range=point_cloud_range,
        anchor_generator_config=anchor_generator_cfg
    )
    A.generate_anchors(feature_map_size)
