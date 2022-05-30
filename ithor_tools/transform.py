import numpy as np
import math

import open3d as o3d
import matplotlib.pyplot as plt
from skimage.transform import resize
import copy

def cornerpoint_projection(cornerpoints):
    res = []
    for e,c in enumerate(cornerpoints):
        if e%4==0 or e%4==1:
            res.append([c[0],c[2]])
    return res

def to_rad(th):
    return th*math.pi / 180

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

class attn2map():
    def __init__(self,metric_map):
        self.metric_map = metric_map
        self.new_map = np.zeros_like(metric_map)
        width = 800
        height = 800
        fov = 60
        self.width = width
        self.height = height
        # camera intrinsics
        focal_length = 0.5 * width / math.tan(to_rad(fov/2))
        fx, fy, cx, cy = (focal_length,focal_length, width/2, height/2)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 
                                fx, fy, cx, cy)
                                
    def get_projection(self,controller,attn,agent_pos,rot):
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
        
        rot = math.pi*rot/180
        pcd.transform([[-math.sin(rot), 0,-math.cos(rot), agent_pos['z']],
                [0, 1, 0, agent_pos['y']],
                [math.cos(rot), 0, -math.sin(rot), agent_pos['x']],
                [0, 0, 0, 1]])

        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                    voxel_size=0.1)
        # print('voxelization')
        # o3d.visualization.draw_geometries([voxel_grid])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        indices = np.stack(list(vx.grid_index for vx in voxels))
        shape = indices.max(axis=0)
        res = np.zeros((shape[2]+1,shape[0]+1))
        vpos= []
        for vx in voxels:
            grid_index = vx.grid_index
            pos_temp = voxel_grid.get_voxel_center_coordinate(grid_index)
            color = vx.color[0]
            temp = res[grid_index[2],grid_index[0]]
            if color>temp and color>0.75:
                res[grid_index[2],grid_index[0]] = color
                vpos.append(dict(pos=pos_temp,color = color))
        del pcd, voxel_grid
        return res,vpos

    def flip(self,res,rot):
        if rot == 270:
            return np.flip(res,axis=0)

    def transform(self,attn,controller,scenemap):
        agent_pos = controller.last_event.metadata['agent']['position']
        agent_rot = controller.last_event.metadata['agent']['rotation']['y']
        res,vpos = self.get_projection(controller,attn,agent_pos,agent_rot)
        for voxel in vpos:
            pos = voxel['pos']
            pos = dict(x=pos[2],y=pos[1],z=pos[0])
            pos = scenemap.xyz2grid(pos)

            self.new_map[pos[0],pos[1]] = voxel['color']
        
        new_map = copy.deepcopy(self.new_map)
        new_map = np.rot90(new_map)

        map = copy.deepcopy(self.metric_map)
        map = np.rot90(map)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(new_map)
        plt.subplot(1,2,2)
        plt.imshow(map)
        plt.plot()
        del new_map,map
    
    def reset(self):
        self.new_map = np.zeros_like(self.metric_map)