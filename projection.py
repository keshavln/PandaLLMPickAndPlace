import numpy as np
import cv2 as cv
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix

def find_xyz(env, obs, boxes, CAMERA_WIDTH=640, CAMERA_HEIGHT=480, to_be_detected='object and target'):
    """
    Finds the 3D coordinates of the objects in the camera's frame, using the pinhole camera model.
    Arguments:
    - env : the robosuite environment
    - obs : the observation from the environment
    - boxes : array returned by Grounding DINO containing bounding box coordinates
    - CAMERA_WIDTH : the width of the camera
    - CAMERA_HEIGHT : the height of the camera
    - to_be_detected : can be used to detect just the object, just the target, or both
    """

    K = get_camera_intrinsic_matrix(
        sim=env.sim,
        camera_name="robot0_eye_in_hand",
        camera_height=CAMERA_HEIGHT,
        camera_width=CAMERA_WIDTH
    )

    fx = float(K[0,0])
    fy = float(K[1,1])

    cx = CAMERA_WIDTH/2
    cy = CAMERA_HEIGHT/2

    depth = obs["robot0_eye_in_hand_depth"]
    real_depth = get_real_depth_map(env.sim, depth)
    real_depth = np.squeeze(real_depth)

    outputs = []

    if 'object' in to_be_detected:
        object_u = int(float(boxes[0][0]) * CAMERA_WIDTH)
        object_v = int(float(boxes[0][1]) * CAMERA_HEIGHT)
        Z_object = float(real_depth[object_v, object_u])
        X_object = (object_u - cx) * Z_object / fx
        Y_object = (object_v - cy) * Z_object / fy
        outputs += [X_object, Y_object, Z_object]

    if 'target' in to_be_detected:
        target_u = int(float(boxes[1][0]) * CAMERA_WIDTH)
        target_v = int(float(boxes[1][1]) * CAMERA_HEIGHT)
        Z_target = float(real_depth[target_v, target_u])
        X_target = (target_u - cx) * Z_target / fx
        Y_target = (target_v - cy) * Z_target / fy
        outputs += [X_target, Y_target, Z_target]

    return outputs