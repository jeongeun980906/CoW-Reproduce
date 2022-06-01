import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import math
import cv2
import copy
from FBE.union import UnionFind
                
def to_rad(th):
    return th*math.pi / 180
    
class gridmap():
    def __init__(self,scenebound,stepsize=0.1):
        '''
        All of the unknown first
        '''

        self.robot_size = 4
        scenebound = np.asarray(scenebound)
        x_max, z_max = np.max(scenebound,axis=0)
        x_min, z_min  = np.min(scenebound,axis=0)
        print(x_min,x_max,z_min,z_max)
        self.stepsize = stepsize
        x_max = self.stepsize* (x_max//self.stepsize)
        z_max = self.stepsize* (z_max//self.stepsize)
        x_min = self.stepsize* (x_min//self.stepsize +1)
        z_min = self.stepsize* (z_min//self.stepsize +1)

        x_len =  x_max- x_min
        z_len =  z_max- z_min
        # print(x_min,x_max,z_min,z_max)
        self.x_min, self.x_max = x_min, x_max
        self.z_min, self.z_max = z_min, z_max
        self.y_default = 0.91
        w_quan = int(x_len//self.stepsize)+1
        h_quan = int(z_len//self.stepsize)+1
        
        self.w_quan = w_quan
        self.h_quan = h_quan
        
        self.map = np.ones((w_quan,h_quan,3))/2
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
                                

    def scan_single(self,controller,temp_map):
        agent_pos = controller.last_event.metadata['agent']['position']
        agent_rot = controller.last_event.metadata['agent']['rotation']
        DEPTH = controller.last_event.depth_frame
        COLOR = controller.last_event.frame
        GRAY = (np.sum(COLOR,axis=-1)/(3*255)).astype(np.float32)
        depth = o3d.geometry.Image(DEPTH)
        color = o3d.geometry.Image(GRAY)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=2.0,
                                                                    convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        rot = math.pi*agent_rot['y']/180
        pcd.transform([[-math.sin(rot), 0,-math.cos(rot), agent_pos['z']],
                [0, 1, 0, agent_pos['y']],
                [math.cos(rot), 0, -math.sin(rot), agent_pos['x']],
                [0, 0, 0, 1]])

        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                    voxel_size=self.stepsize)
        # print('voxelization')
        # o3d.visualization.draw_geometries([voxel_grid])

        voxels = voxel_grid.get_voxels()  # returns list of voxels
        min_bound = voxel_grid.get_min_bound()
        max_bound = voxel_grid.get_max_bound()
        min_bound = dict(x=min_bound[2],y=min_bound[1],z=min_bound[0])
        max_bound = dict(x=max_bound[2],y=max_bound[1],z=max_bound[0])
        indices = np.stack(list(vx.grid_index for vx in voxels))
        shape = indices.max(axis=0)
        res = np.zeros((shape[2]+1,shape[0]+1))
        # print(max_map_pos,min_map_pos)
        for vx in voxels:
            grid_index = vx.grid_index
            pos_temp = voxel_grid.get_voxel_center_coordinate(grid_index)
            if grid_index[1] != 0 and grid_index[1]!= shape[1] and pos_temp[1]<1.2: # ceiling?
                res[grid_index[2],grid_index[0]] = 1
                map_pos = dict(x=pos_temp[2],y=pos_temp[1],z=pos_temp[0])
                map_pos = self.xyz2grid(map_pos)
                temp_map[map_pos[0],map_pos[1],:] = [0,0,0]
        # plt.figure()
        # plt.imshow(res)
        # plt.plot()
        return temp_map,[self.xyz2grid(min_bound),self.xyz2grid(max_bound)]

    def scan_full(self,controller):
        pos = controller.last_event.metadata['agent']['position']
        bounds = []
        temp_map = np.ones((self.w_quan,self.h_quan,3))/2
        grid_pos = self.xyz2grid(pos)
        temp_map, bound = self.scan_single(controller,temp_map)
        bounds.append(bound)
        temp_map[grid_pos[0],grid_pos[1]] = [1,1,1]
        for _ in range(5):
            controller.step(
                    action="RotateRight",degrees = 60
                )
            temp_map,bound = self.scan_single(controller,temp_map)
            bounds.append(bound)
        controller.step(
                    action="RotateRight",degrees = 60
                )
        for i in np.arange(start=-1,stop=1.01,step=0.01):
            temp_map = self.ray_x(grid_pos,temp_map,i) 
            temp_map = self.ray_y(grid_pos,temp_map,i) 
        for bound in bounds:
            for i in range(bound[0][0],bound[1][0]):
                for j in range(bound[0][1],bound[1][1]):
                    if temp_map[i,j,0] == 0.5:
                        check_left = temp_map[i-1,j,0] == 0
                        check_right = temp_map[i+1,j,0] == 0
                        check_down = temp_map[i,j+1,0] == 0
                        check_up = temp_map[i,j-1,0] == 0
                        if check_right and check_left and check_down and check_up:
                            temp_map[i,j] = [0,0,0]
                        else:
                            check_left_2 = int(temp_map[i-1,j,0] != 0.5)
                            check_right_2 = int(temp_map[i+1,j,0] != 0.5)
                            check_down_2 = int(temp_map[i,j+1,0] != 0.5)
                            check_up_2 = int(temp_map[i,j-1,0] != 0.5)
                            if (check_right_2 + check_left_2 + check_down_2 + check_up_2)>2:
                                temp_map[i,j] = [1,1,1]

        self.merge_map(temp_map)
        return temp_map

    def frontier_detection(self,cpos):
        img_gray = copy.deepcopy(self.map)
        img_gray = cv2.cvtColor(img_gray.astype(np.float32), cv2.COLOR_BGR2GRAY)
        img_gray_recolor = np.where(img_gray==0.5,0.1,img_gray)
        img_gray_recolor = np.where(img_gray==1,0.9,img_gray_recolor)
        img_gray_recolor = np.where(img_gray==0,0.1,img_gray_recolor)
        img_gray_recolor = (img_gray_recolor*255).astype(np.uint8)
        # img_gray_recolor = cv2.resize(img_gray_recolor,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        edges = cv2.Canny(img_gray_recolor,20,10)

        frontier_map = np.zeros_like(img_gray)
        index = np.where(edges != 0)
        res = []
        for indx in zip(index[0],index[1]):
            left = self.map[indx[0]-1,indx[1],0]==0
            right = self.map[indx[0]+1,indx[1],0]==0
            up = self.map[indx[0],indx[1]+1,0]==0
            down = self.map[indx[0],indx[1]-1,0] == 0
            center = self.map[indx[0],indx[1],0]==0
            if left+right+up+down+center ==0:
                frontier_map[indx[0],indx[1]]=1
                res.append(indx)
        
        groups = self.groupTPL(res)
        filter_by_size = []
        distances = []
        for group in groups:
            if len(group)>self.robot_size:
                mean_x = sum([x[0] for x in group])/len(group)
                mean_y = sum([y[1] for y in group])/len(group)
                frontier_map[int(mean_x),int(mean_y)] = 0.5
                mean = [int(mean_x),int(mean_y)]
                mean = self.grid2xyz(mean)
                dis = self.get_distance(cpos,mean)
                distances.append(dis)
                filter_by_size.append(mean)
        if len(distances)>0:
            sort_index = np.argsort(np.asarray(distances))
            return filter_by_size,sort_index
        else:
            return [],None

    def groupTPL(self,TPL, distance=1):
        U = UnionFind()

        for (i, x) in enumerate(TPL):
            for j in range(i + 1, len(TPL)):
                y = TPL[j]
                if max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= distance:
                    U.union(x, y)

        disjSets = {}
        for x in TPL:
            s = disjSets.get(U[x], set())
            s.add(x)
            disjSets[U[x]] = s

        return [list(x) for x in disjSets.values()]

    def ray_x(self,start_pos,temp_map,angle): # -1~1
        sign = [1,2] if angle<0 else [-1,0]
        x_ = np.arange(start_pos[0]+1,start_pos[0]+2/self.stepsize).astype(np.int16)
        y_ = ((angle*(x_-start_pos[0]))+start_pos[1]).astype(np.int16)
        # if sum(temp_map[x_[0],y_[0]+sign[0]:y_[0]+sign[1],0]==0)==0:
        for x,y in zip(x_,y_) or sum(temp_map[x+1:x+2,y,0]==0)>0:
            if temp_map[x,y,0]==0:
                break
            else:
                temp_map[x,y] = [1,1,1]
                
        x_ = np.arange(start_pos[0]-1,start_pos[0]-2/self.stepsize,-1).astype(np.int16)
        y_ = ((angle*(x_-start_pos[0]))+start_pos[1]).astype(np.int16)
        if  sum(temp_map[x_[0],y_[0]+1:y_[0]+2,0]==0)==0:
            for x,y in zip(x_,y_) or sum(temp_map[x-1:x,y,0]==0)>0:
                if temp_map[x,y,0]==0:
                    break
                else:
                    temp_map[x,y] = [1,1,1]
        return temp_map

    def ray_y(self,start_pos,temp_map,angle): # -1~1
        y_ = np.arange(start_pos[1]+1,start_pos[1]+2/self.stepsize).astype(np.int16)
        x_ = ((angle*(y_-start_pos[1]))+start_pos[0]).astype(np.int16)
        if  sum(temp_map[x_[0]+1:x_[0]+2,y_[0],0]==0)==0:
            for x,y in zip(x_,y_):
                if temp_map[x,y,0]==0 or sum(temp_map[x,y-1:y,0]==0)>0: #(temp_map[x,y-1,0]==0 and sum(temp_map[x-1:x+3,y,0]))>1:
                    break
                else:
                    temp_map[x,y] = [1,1,1]

        y_ = np.arange(start_pos[1]-1,start_pos[1]-2/self.stepsize,-1).astype(np.int16)
        x_ = ((angle*(y_-start_pos[1]))+start_pos[0]).astype(np.int16)
        for x,y in zip(x_,y_):
            if temp_map[x,y,0]==0 or sum(temp_map[x,y+1:y+2,0]==0)>0:
                break
            else:
                temp_map[x,y] = [1,1,1]
        return temp_map

    def merge_map(self,temp_map):
        occupied = np.where(temp_map==0)
        self.map[occupied] =0
        free = (temp_map==1) * (self.map>0)
        self.map[free] = 1
        for i in range(1,self.map.shape[0]-1):
            for j in range(1,self.map.shape[1]-1):
                check_left = self.map[i-1,j,0] == 0
                check_right = self.map[i+1,j,0] == 0
                check_down = self.map[i,j+1,0] == 0
                check_up = self.map[i,j-1,0] == 0
                if check_right and check_left and check_down and check_up:
                    self.map[i,j] = [0,0,0]

        

    def xyz2grid(self,pos):
        x = pos['x']
        z = pos['z']
        w = int((x - self.x_min)//self.stepsize)
        h = int((z - self.z_min)//self.stepsize)
        return [w,h]
        
    def grid2xyz(self,wh,y=None):
        if y==None:
            y=self.y_default
        x = wh[0] * self.stepsize + self.x_min

        z = wh[1] * self.stepsize + self.z_min
        
        return dict(x=x,y=y,z=z)
    def get_distance(self,pos1,pos2):
        return math.sqrt((pos1['x']-pos2['x'])**2+(pos1['z'] - pos2['z'])**2)