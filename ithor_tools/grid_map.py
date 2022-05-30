import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import math

class gridmap():
    def __init__(self,gt_map):
        self.map = np.zeros_like(gt_map)

    def scan(self,controller):
        agent_pos = controller.last_event.metadata['agnet']['position']
        agent_rot = controller.last_event.metadata['agnet']['rotation']
        DEPTH = controller.last_event.depth_frame
        COLOR = controller.last_event.frame

        attn = 10*attn.astype(np.float32)
        attn = np.clip(attn,0,1)

        depth = o3d.geometry.Image(DEPTH)
        color = o3d.geometry.Image(attn)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=5.0,
                                                                    convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        rot = math.pi*agent_rot['y']/180
        pcd.transform([[-math.sin(rot), 0,-math.cos(rot), agent_pos['z']],
                [0, 1, 0, agent_pos['y']],
                [math.cos(rot), 0, -math.sin(rot), agent_pos['x']],
                [0, 0, 0, 1]])

        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                    voxel_size=0.1)
        # print('voxelization')
        o3d.visualization.draw_geometries([voxel_grid])