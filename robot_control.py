import numpy as np
import cv2 as cv

def move_to_pose_absolute(env, initial_pos, final_pos, gripper_state, max_steps=200, kp=0.7):
    """
    Moves the end-effector to the specified absolute position using proportional control.
    Arguments:
    - env : the robosuite environment
    - initial_pos : the initial position of the end-effector
    - final_pos : the final position of the end-effector
    - gripper_state : -1.0 for open, 1.0 for closed
    - max_steps : the maximum number of steps, after which the movement automatically terminates
    - kp : proportional gain
    """
    error = final_pos - initial_pos
    step_count = 1
    while np.linalg.norm(error) > 0.015 and step_count < max_steps:

        p_correction = kp * error
        #print(p_correction)

        action = np.zeros(7)
        action[:3] = p_correction
        action[6] = gripper_state

        running_obs, reward, done, info = env.step(action)
        env.render()

        rgb_image = running_obs["robot0_eye_in_hand_image"]
        bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        cv.imshow('live feed', bgr_image)
        cv.waitKey(1)

        running_pos = running_obs['robot0_eef_pos']
        error = final_pos - running_pos

        step_count += 1

    return running_obs

def move_to_pose_delta(env, init_obs, x, y, z, kp=0.7, max_steps=200, x_tol=0.1, z_tol=0.1, gripper_state=-1.0):
    """
    Moves the end-effector by a certain delta in the x, y and z directions in the camera's frame.
    """
    world_x = y
    world_y = -x
    world_z = -z

    start_pos = init_obs['robot0_eef_pos']
    target_pos = start_pos + np.array([world_x+x_tol,world_y,world_z+z_tol])
    current_obs = init_obs
    error = target_pos - start_pos
    step_count = 1

    while np.linalg.norm(error) > 0.007 and step_count < max_steps:
        current_pos = current_obs['robot0_eef_pos']
        error = target_pos - current_pos
        p_correction = kp * error

        action = np.zeros(7)
        action[:3] = p_correction
        action[6] = gripper_state

        current_obs, reward, done, info = env.step(action)
        env.render()

        rgb_image = current_obs["robot0_eye_in_hand_image"]
        bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        cv.imshow('live feed', bgr_image)
        cv.waitKey(1)
        step_count += 1
    
    return current_obs