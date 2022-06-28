import ai2thor
from ai2thor.controller import Controller,BFSController
from ai2thor.platform import CloudRendering
import torch
import math
from tqdm import tqdm
from ithor_tools.vis_tool import *
from ithor_tools.transform import cornerpoint_projection,attn2map
from ithor_tools.map import single_scenemap
from grad_cam.cam import clip_grad_cam
from FBE.fbe import gridmap
from ai2thor.util.metrics import get_shortest_path_to_object
from ithor_tools.objects import choose_query_objects
# Planning Module
from RRT import gridmaprrt as rrt
from FBE.step import step_frontier
from ithor_tools.store import score_storage
import random


RESUME = False
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_gradcam = clip_grad_cam(device)
ST = score_storage()
if RESUME:
    ST.load_json()
# scene_names = ['FloorPlan_Train8_1','FloorPlan_Train1_2', 'FloorPlan_Train10_4','FloorPlan_Train9_1','FloorPlan_Train6_2',
#                 'FloorPlan_Train2_3','FloorPlan_Train5_1']

val = [f"FloorPlan_Val{i}_{j}" for i in range(1,4) for j in range(1,6)]
# val = ["FloorPlan_Val3_2","FloorPlan_Val3_3","FloorPlan_Val3_4","FloorPlan_Val3_4"]
scene_names = val
# scene_names = scene_names[9:]
for scene_name in scene_names:
    gridSize=0.05

    controller = Controller(
        platform = CloudRendering,
        agentMode="locobot",
        visibilityDistance=2.5,
        scene = scene_name,
        gridSize=gridSize,
        movementGaussianSigma=0,
        rotateStepDegrees=90,
        rotateGaussianSigma=0,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=300,
        height=300,
        fieldOfView=90
    )

    controller.reset(
        # makes the images a bit higher quality
        width=800,
        height=800,

        # Renders several new image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderNormalsImage=False
    )
    scene_bounds = controller.last_event.metadata['sceneBounds']['center']

    controller.step(
        action="AddThirdPartyCamera",
        position=dict(x=scene_bounds['x'], y=5.0, z=scene_bounds['z']),
        rotation=dict(x=90, y=0, z=0),
        orthographic=True,
        orthographicSize= 5.0, fieldOfView=100,
        skyboxColor="white"
    )
    controller.step(dict(action='GetReachablePositions'))
    rstate = controller.last_event.metadata['actionReturn']

    objects = controller.last_event.metadata['objects']
    query_objects = choose_query_objects(objects,'all')
    rrtplanner = rrt.RRT(controller = controller, expand_dis=0.1,max_iter=10000,goal_sample_rate=20)

    controller.step(
        action="Teleport",
        position = rstate[100],
        rotation = dict(x=0,y=270,z=0)
    )

    scene_bounds = controller.last_event.metadata['sceneBounds']['cornerPoints']
    scene_bounds = cornerpoint_projection(scene_bounds)
    sm = single_scenemap(scene_bounds,rstate,stepsize = 0.1)

    proj = attn2map(controller,sm.gridmap)
    fbe = gridmap(controller,scene_bounds,clip_gradcam,proj,sm)
    for query_object in tqdm(query_objects,desc = scene_name):
        proj.reset()
        fbe.reset()

        controller.step(
            action="Teleport",
            position = rstate[100],
            rotation = dict(x=0,y=270,z=0)
             )

        pos = controller.last_event.metadata['agent']['position']
        try:
            min_path = get_shortest_path_to_object(controller,query_object['objectId'],pos)
            min_length = 0
            last_pos = pos
            for p in min_path:
                min_length += math.sqrt((last_pos['x']-p['x'])**2+(last_pos['z']-p['z'])**2)
                last_pos = p
        except:
            print('path error')
            min_length = 100
        if min_length==0:
            min_length+=0.1

        query_object_name = query_object['objectType']

        new_query_object_name = ''
        if len(query_object_name)>2:
            for i, letter in enumerate(query_object_name):
                if i and letter.isupper():
                    new_query_object_name += ' '
                new_query_object_name += letter.lower()
        else:
            new_query_object_name = query_object_name
        clip_gradcam.set_text(new_query_object_name)
        
        sucess = False
        total_path_len = 0
        flag = 0
        total_path_len = 0
        while sucess != True:
            cpos = controller.last_event.metadata['agent']['position']
            gt_find,find,path_len, reset_map  = step_frontier(fbe,rrtplanner,controller,query_object)
            sucess = gt_find*find
            total_path_len += path_len
            npos = controller.last_event.metadata['agent']['position']
            if math.sqrt((cpos['x']-npos['x'])**2+(cpos['z']-npos['z'])**2) < 0.1:
                reset_map = True
                flag +=1
            else:
                flag = 0
            if flag>4:
                break
            if reset_map:
                proj.reset()
                fbe.reset()
            if total_path_len>50:
                break
        SPL = sucess*min_length/total_path_len
        ST.append(SPL,query_object_name,scene_name)
        ST.save_json()
    controller.stop()
df = ST.average()
print(df)
