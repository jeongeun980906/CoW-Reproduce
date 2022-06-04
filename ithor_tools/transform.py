import numpy as np
import math

import open3d as o3d
import matplotlib.pyplot as plt
from skimage.transform import resize
import copy

def ndarray(list):
    array = np.array(list)
    return array


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
    def __init__(self,controller,metric_map):
        self.controller = controller
        self.metric_map = copy.deepcopy(metric_map)
        self.new_map = 1- np.expand_dims(copy.deepcopy(metric_map),axis=-1)
        self.new_map = np.repeat(self.new_map,3,axis=-1)
        width = 800
        height = 800
        fov = 90
        self.width = width
        self.height = height
        # camera intrinsics
        focal_length = 0.5 * width / math.tan(to_rad(fov/2))
        fx, fy, cx, cy = (focal_length,focal_length, width/2, height/2)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 
                                fx, fy, cx, cy)
                                
    def get_projection(self,attn,agent_pos,rot):
        DEPTH = self.controller.last_event.depth_frame
        COLOR = self.controller.last_event.frame

        attn = 15*attn.astype(np.float32)
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

    def transform(self,attn,scenemap,vis=True):
        agent_pos = self.controller.last_event.metadata['agent']['position']
        agent_rot = self.controller.last_event.metadata['agent']['rotation']['y']
        _,vpos = self.get_projection(attn,agent_pos,agent_rot)
        for voxel in vpos:
            pos = voxel['pos']
            pos = dict(x=pos[2],y=pos[1],z=pos[0])
            pos = scenemap.xyz2grid(pos)
            color = (voxel['color']-0.75)/0.25 # 0~1
            self.new_map[pos[0],pos[1]] = [color,(color*(1-color))/2,1- color]
        if vis:
            new_map = copy.deepcopy(self.new_map)
            new_map = np.rot90(new_map)

            plt.figure()
            plt.imshow(new_map)
            plt.axis("off")
            plt.show()
            del new_map
        if len(vpos)>0:
            return True
        else:
            return False
    
    def reset(self):
        self.new_map = 1- np.expand_dims(copy.deepcopy(self.metric_map),axis=-1)
        self.new_map = np.repeat(self.new_map,3,axis=-1)