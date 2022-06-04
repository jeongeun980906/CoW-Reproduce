import numpy as np
import matplotlib.pyplot as plt

def step_local_search(controller,clip_gradcam,proj,scenemap):
    # for i in range(6):
    #     controller.step(
    #         action = 'RotateRight', degrees = 60
    #     )
    img= controller.last_event.frame
    attn,_ = clip_gradcam.run(img,vis=False)
    find = proj.transform(attn,scenemap,vis=False)
    del attn, img
    return find

def check_vis(controller,query_object_ID,show=False):
    instance_segmentation = controller.last_event.instance_segmentation_frame
    obj_colors = controller.last_event.object_id_to_color
    temp = np.zeros((instance_segmentation.shape[0],instance_segmentation.shape[1]))    
    query_color = obj_colors[query_object_ID]
    
    # print(controller.last_event.object_id_to_color)
    R = (instance_segmentation[:,:,0]==query_color[0])
    G = (instance_segmentation[:,:,1]==query_color[1])
    B = (instance_segmentation[:,:,2]==query_color[2])
    total = R & G & B
    temp[total] = +1
    if show:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(controller.last_event.frame)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(temp)
        plt.axis('off')
        plt.show()

    # thres = np.max(temp)
    # if thres < 3:
    #     thres = 3
    
    temp = np.where(temp>=1)
    if len(temp[0])>0:
        GT_box = [min(temp[1]),min(temp[0]),max(temp[1]),max(temp[0])]
        area = (GT_box[2]-GT_box[0])*(GT_box[3]-GT_box[1])
        # print(area/((instance_segmentation.shape[0]*instance_segmentation.shape[1])))
        if area>0.01*(instance_segmentation.shape[0]*instance_segmentation.shape[1]):
            del instance_segmentation,obj_colors
            return True
        else:
            del instance_segmentation,obj_colors
            return False
    else:
        del instance_segmentation,obj_colors
        return False