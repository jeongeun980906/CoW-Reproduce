from RRT import gridmaprrt_pathsmoothing as smoothing
from ithor_tools.utils import check_vis,step_local_search
import numpy as np
import math

def step_frontier(fbe,rrtplanner,controller,clip_gradcam,proj,scenemap,query_object):
    _,gt_find,find= fbe.scan_full(clip_gradcam,proj,scenemap,query_object['objectId'])
    if gt_find:
        img= controller.last_event.frame
        _,_ = clip_gradcam.run(img,vis=True)
        return gt_find,find,0.1, False
    cpos = controller.last_event.metadata['agent']['position']
    waypoints, indexs = fbe.frontier_detection(cpos)
    grid_cpos = fbe.xyz2grid(cpos)
    fbe.map[grid_cpos[0],grid_cpos[1]] = [0.7,0.7,0]
    # imshow_grid = sm.plot(cpos)
    # plot_frames(controller.last_event,imshow_grid)
    print(waypoints)
    if len(waypoints)> 0:
        rrtplanner.set_start(cpos)
        rrtplanner.set_goal(waypoints[indexs[0]])
        print("start planning")
        local_path = rrtplanner.planning(animation=False)
        try:
            smoothpath = smoothing.path_smoothing(rrtplanner,40,verbose=False)
        except:
            smoothpath = local_path
        # rrtplanner.plot_path(smoothpath)
        fr_pos = rrtplanner.rstate[smoothpath[0]]
        total_path_len = 0
        for p in smoothpath[1:]:
            to_pos = rrtplanner.rstate[p]
            delta = to_pos - fr_pos
            d = np.linalg.norm(delta)
            total_path_len += d
            theta = (math.atan2(-delta[1],delta[0])/math.pi*180+90)
            # print(theta)
            controller.step(
                action="Teleport",
                position = dict(x=to_pos[0],y=0.91,z=to_pos[1]),
                rotation = dict(x=0,y=theta,z=0)
            )
            fr_pos = to_pos
            find = step_local_search(controller,clip_gradcam,proj,scenemap)
            gt_find = check_vis(controller, query_object['objectId'],False)
            if gt_find:
                img= controller.last_event.frame
                _,_ = clip_gradcam.run(img,vis=True)
                return gt_find,find,total_path_len, False
            action_sucess = controller.last_event.metadata['lastActionSuccess']
            if not action_sucess:
                grid_pos = fbe.xyz2grid(dict(x=to_pos[0],y=0.91,z=to_pos[1]))
                fbe.map[grid_pos[0],grid_pos[1]] = [0,0.7,1]
                break
        grid_pos = fbe.xyz2grid(waypoints[indexs[0]])
        fbe.map[grid_pos[0],grid_pos[1]] = [0,0.7,1]
        fbe.map[grid_cpos[0],grid_cpos[1]] = [1,1,1]
        fbe.map[grid_pos[0],grid_pos[1]] = [1,1,1]
    else:
        print("Done Exploration")
        return False, False,0, True

    return False,False,total_path_len, False