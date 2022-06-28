from RRT import gridmaprrt_pathsmoothing as smoothing
from ithor_tools.utils import check_vis
import numpy as np
import math
import matplotlib.pyplot as plt

def step_frontier(fbe,rrtplanner,controller,query_object,clip_gradcam=None,vis=False,verbose=False):
    temp_map,local_goals,gt_find,find= fbe.scan_full(query_object)
    if gt_find:
        if vis:
            img= controller.last_event.frame
            _,_ = clip_gradcam.run(img,vis=True)
        if find:
            return gt_find,find,0.1, False
    cpos = controller.last_event.metadata['agent']['position']
    if len(local_goals) == 0:
        waypoints, indexs = fbe.frontier_detection(cpos)
        if len(waypoints)> 0:
            indexs = [indexs[0]]
            
    else:
        waypoints = local_goals
        indexs = [i for i in range(len(local_goals))]
    # imshow_grid = sm.plot(cpos)
    # plot_frames(controller.last_event,imshow_grid)
    if verbose:
        print(waypoints)
    if len(waypoints)> 0:
        for index in indexs:
            cpos = controller.last_event.metadata['agent']['position']
            grid_cpos = fbe.xyz2grid(cpos)
            fbe.map[grid_cpos[0],grid_cpos[1]] = [0.7,0.7,0]
            
            rrtplanner.set_start(cpos)
            rrtplanner.set_goal(waypoints[index])
            if verbose:
                print("start planning")
            local_path = rrtplanner.planning(animation=False)
            try:
                smoothpath = smoothing.path_smoothing(rrtplanner,40,verbose=False)
            except:
                smoothpath = local_path
            if vis:
                rrtplanner.plot_path(smoothpath)
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
                find,_ = fbe.step_local_search()
                gt_find = check_vis(controller, query_object,False)
                if gt_find:
                    if vis:
                        img= controller.last_event.frame
                        _,_ = clip_gradcam.run(img,vis=True)
                    if find:
                        return gt_find,find,total_path_len, False
                action_sucess = controller.last_event.metadata['lastActionSuccess']
                if not action_sucess:
                    grid_pos = fbe.xyz2grid(dict(x=to_pos[0],y=0.91,z=to_pos[1]))
                    fbe.map[grid_pos[0],grid_pos[1]] = [0,0.7,1]
                    break
            grid_pos = fbe.xyz2grid(waypoints[indexs[0]])
            fbe.map[grid_pos[0],grid_pos[1]] = [0,0.7,1]
    else:
        if verbose:
            print("Done Exploration")
        return False, False,0, True
    if vis:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(fbe.map)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(temp_map)
        plt.axis('off')
        plt.show()
    fbe.map[grid_cpos[0],grid_cpos[1]] = [1,1,1]
    fbe.map[grid_pos[0],grid_pos[1]] = [1,1,1]
    return False,False,total_path_len, False