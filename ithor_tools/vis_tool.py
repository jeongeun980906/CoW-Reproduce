import sys
from typing import Sequence
import numpy as np
import os
from typing import Optional
import ai2thor.server
from typing import Union
import seaborn as sns
import cv2
import copy
import matplotlib.pyplot as plt

def plot_frames(event: Union[ai2thor.server.Event, np.ndarray],gridmap : np.ndarray) -> None:
    """Visualize all the frames on an AI2-THOR Event.
    Example:
    plot_frames(controller.last_event)
    """
    if isinstance(event, ai2thor.server.Event):
        third_person_frames = event.third_party_camera_frames
        RGB = event.frame
        DEPTH = event.depth_frame

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(7, 14))
        grid = plt.GridSpec(13, 6, wspace=0.4, hspace=0.3)

        ax = fig.add_subplot(grid[:4, :3])
        im = ax.imshow(RGB)
        ax.axis("off")
        ax.set_title('RGB')

        ax = fig.add_subplot(grid[:4, 3:])
        im = ax.imshow(DEPTH)
        ax.axis("off")
        ax.set_title('DEPTH')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        # add third party camera frames
        ax = fig.add_subplot(grid[4:8, :])
        ax.set_title("Map View")
        temp = crop_zeros(third_person_frames[0])
        print(temp.shape)
        ax.imshow(temp)
        ax.axis("off")

        ax = fig.add_subplot(grid[8:12, :])
        ax.set_title("Grid Map")
        ax.imshow(gridmap,cmap=plt.cm.gray_r)
        ax.axis("off")

        # plt.tight_layout()
        plt.show()


def crop_zeros(image):
    y_nonzero, x_nonzero, _ = np.nonzero(1-image/255)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def show_objects_table(objects: list) -> None:
    """Visualizes objects in a way that they are clickable and filterable.
    Example:
    event = controller.step("MoveAhead")
    objects = event.metadata["objects"]
    show_objects_table(objects)
    """
    import pandas as pd
    from collections import OrderedDict

    processed_objects = []
    for obj in objects:
        obj = obj.copy()
        obj["position[x]"] = round(obj["position"]["x"], 4)
        obj["position[y]"] = round(obj["position"]["y"], 4)
        obj["position[z]"] = round(obj["position"]["z"], 4)

        obj["rotation[x]"] = round(obj["rotation"]["x"], 4)
        obj["rotation[y]"] = round(obj["rotation"]["y"], 4)
        obj["rotation[z]"] = round(obj["rotation"]["z"], 4)

        del obj["position"]
        del obj["rotation"]

        # these are too long to display
        del obj["objectOrientedBoundingBox"]
        del obj["axisAlignedBoundingBox"]
        del obj["receptacleObjectIds"]

        obj["mass"] = round(obj["mass"], 4)
        obj["distance"] = round(obj["distance"], 4)

        obj = OrderedDict(obj)
        obj.move_to_end("distance", last=False)
        obj.move_to_end("rotation[z]", last=False)
        obj.move_to_end("rotation[y]", last=False)
        obj.move_to_end("rotation[x]", last=False)

        obj.move_to_end("position[z]", last=False)
        obj.move_to_end("position[y]", last=False)
        obj.move_to_end("position[x]", last=False)

        obj.move_to_end("name", last=False)
        obj.move_to_end("objectId", last=False)
        obj.move_to_end("objectType", last=False)

        processed_objects.append(obj)

    df = pd.DataFrame(processed_objects)
    print(
        "Object Metadata. Not showing objectOrientedBoundingBox, axisAlignedBoundingBox, and receptacleObjectIds for clarity."
    )
    pd.set_option('display.max_rows', None)
    return df