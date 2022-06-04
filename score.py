import ai2thor
from ai2thor.controller import Controller,BFSController
from ai2thor.platform import CloudRendering
import torch
import math

from ithor_tools.vis_tool import *
from ithor_tools.transform import cornerpoint_projection,attn2map
from ithor_tools.map import single_scenemap
from grad_cam.cam import clip_grad_cam
from FBE.fbe import gridmap
from ai2thor.util.metrics import get_shortest_path_to_object
from ithor_tools.utils import check_vis,step_local_search
# Planning Module
from RRT import gridmaprrt as rrt
from FBE.step import step_frontier

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_gradcam = clip_grad_cam(device)

scene_name = 'FloorPlan_Train1_2'
gridSize=0.05

controller = Controller(
    agentMode="locobot",
    visibilityDistance=2.0,
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

controller.step(
    action="Teleport",
    position = rstate[200],
    rotation = dict(x=0,y=270,z=0)
)

objects = controller.last_event.metadata['objects']
query_object_index = 16
query_object = objects[query_object_index]
pos = controller.last_event.metadata['agent']['position']
try:
    min_path = get_shortest_path_to_object(controller,query_object['objectId'],pos)
    min_length = 0
    last_pos = pos
    for p in min_path:
        min_length += math.sqrt((last_pos['x']-p['x'])**2+(last_pos['z']-p['z'])**2)
        last_pos = p
except:
    min_length = 0.1


controller.step(
    action="Teleport",
    position = rstate[200],
    rotation = dict(x=0,y=270,z=0)
)

scene_bounds = controller.last_event.metadata['sceneBounds']['cornerPoints']
scene_bounds = cornerpoint_projection(scene_bounds)
sm = single_scenemap(scene_bounds,rstate,stepsize = 0.1)

fbe = gridmap(controller,scene_bounds)

proj = attn2map(controller,sm.gridmap)

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

rrtplanner = rrt.RRT(controller = controller, expand_dis=0.1,max_iter=10000,goal_sample_rate=20)
proj.reset()
fbe.reset()
gt_find = False
total_path_len = 0
while gt_find != True:
    gt_find,sucess,path_len, reset_map  = step_frontier(fbe,rrtplanner,controller,clip_gradcam,proj,sm,query_object)
    total_path_len += path_len
    if reset_map:
        proj.reset()
        fbe.reset()
print(sucess, min_length,total_path_len)
print(sucess*(min_length)/total_path_len)