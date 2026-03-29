import numpy as np
import cv2 as cv
import supervision as sv
import tkinter as tk
from tkinter import simpledialog, messagebox

import robosuite
from robosuite.environments.base import register_env
from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from perception import detect_objects
from llm import extract_object_and_target
from projection import find_xyz
from robot_control import move_to_pose_absolute, move_to_pose_delta

@register_env
class VLAPickPlace(Lift):
    """
    Custom environment for a pick and place task with the Panda robotic arm. This environment inherits from robosuite's
    pre-existing Lift environment and adds a few more objects.
    """
    def _load_model(self):
        super()._load_model()

        cube_size=0.02

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        mujoco_arena.set_origin([0, 0, 0])

        self.mujoco_objects = []

        self.cube = BoxObject(
            name="cube",
            size_min=[cube_size, cube_size, cube_size],  # [0.015, 0.015, 0.015],
            size_max=[cube_size, cube_size, cube_size],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            rng=self.rng,
        )

        self.cube2 = BoxObject(
            name="cube2",
            size_min=[cube_size, cube_size, cube_size],  # [0.015, 0.015, 0.015],
            size_max=[cube_size, cube_size, cube_size],  # [0.018, 0.018, 0.018])
            rgba=[0, 0, 1, 1],
            rng=self.rng,
        )

        self.cross = MujocoXMLObject(
            name = "object",
            fname = 'cross.xml'
        )

        self.mujoco_objects = [self.cube, self.cube2, self.cross]

        self.placement_initializer = UniformRandomSampler(
            name = "ObjectSampler",
            mujoco_objects = self.mujoco_objects,
            x_range = [-0.15,0.15],
            y_range = [-0.15,0.15],
            reference_pos = self.table_offset,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            rotation=0.0
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects,
        )

class PickAndPlaceTask():
    """
    This is a completely custom class that defines a full pick-and-place sequence.

    Arguments:
    - prompt : string of the form "<object>,<target>"
    - kp : proportional gain which controls the speed of the end-effector
    - render_detections : can be set to True in order to visualize Grounding DINO's detections
    
    """
    def __init__(self, prompt, CAMERA_WIDTH=640, CAMERA_HEIGHT=480, kp=0.7, render_detections=False):
        """
        The robosuite environment is initialized and initial object & target detections are made.
        The initial estimates of the x,y and z coordinates of the object and target are found using the pinhole camera
        model.
        """
        self.prompt = prompt
        self.object = prompt.split(',')[0]
        self.target = prompt.split(',')[1]
        self.CAMERA_WIDTH = CAMERA_WIDTH
        self.CAMERA_HEIGHT = CAMERA_HEIGHT
        self.kp = kp
        self.gripper_state = -1.0
        self.render_detections = render_detections

        self.first_iter_object = True
        self.first_iter_target = True

        self.env = robosuite.make(
            env_name="VLAPickPlace",
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="robot0_eye_in_hand",
            camera_depths=True,
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            horizon=3000,
            render_camera='agentview'
        )

        self.env.reset()

        action = np.zeros(7)
        action[6] = -1.0
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.env.render()
        self.start_pos = self.obs['robot0_eef_pos']
        self.current_pos = 0

        rgb_image = self.obs["robot0_eye_in_hand_image"]
        annotated_image, boxes = detect_objects(rgb_image, self.prompt)

        if self.render_detections:
            cv.imshow('annotation', annotated_image)
            cv.waitKey(0)

        self.X_object, self.Y_object, self.Z_object, self.X_target, self.Y_target, self.Z_target = find_xyz(self.env, self.obs, boxes)

    def move_to_object(self, x_tol=0.1, z_tol=0.1):
        """
        Moves the end-effector to the object. If this function is called more than once, the object is detected once more
        and the end-effector moves to the object's more accurate recalculated coordinates.
        Arguments:
        - x_tol : tolerance in the x direction (to make sure the object is properly grasped)
        - z_tol : tolerance in the z direction (to make sure the end-effector does not collide with the object from top down)
        """
        if self.first_iter_object:
            print("Moving to object...")

            world_x = self.Y_object
            world_y = -self.X_object
            world_z = -self.Z_object

            target_pos = self.start_pos + np.array([world_x+x_tol,world_y,world_z+z_tol])
            self.obs = move_to_pose_absolute(self.env, self.start_pos, target_pos, self.gripper_state, kp=self.kp)
            self.current_pos = self.obs['robot0_eef_pos']
            self.first_iter_object = False

        else:
            rgb_image = self.obs["robot0_eye_in_hand_image"]
            self.current_pos = self.obs['robot0_eef_pos']
            annotated_image, boxes = detect_objects(rgb_image, self.object)
            self.X_object, self.Y_object, self.Z_object = find_xyz(self.env, self.obs, boxes, to_be_detected='object')

            world_x = self.Y_object
            world_y = -self.X_object
            world_z = -self.Z_object

            target_pos = self.current_pos + np.array([world_x+x_tol,world_y,world_z+z_tol])
            self.obs = move_to_pose_absolute(self.env, self.current_pos, target_pos, self.gripper_state, kp=self.kp)
            self.current_pos = self.obs['robot0_eef_pos']

    def move_to_target(self, x_tol=0.1, z_tol=0.1):
        """
        Moves the end-effector to the target. If this function is called more than once, the target is detected once more
        and the end-effector moves to the target's more accurate recalculated coordinates.
        """
        if self.first_iter_target:
            print("Moving to target...")

            world_x = self.Y_target
            world_y = -self.X_target
            world_z = -self.Z_target

            target_pos = self.start_pos + np.array([world_x+x_tol,world_y,world_z+z_tol])
            self.obs = move_to_pose_absolute(self.env, self.start_pos, target_pos, self.gripper_state, kp=self.kp)
            self.current_pos = self.obs['robot0_eef_pos']
            self.first_iter_target = False

        else:
            rgb_image = self.obs["robot0_eye_in_hand_image"]
            self.current_pos = self.obs['robot0_eef_pos']
            annotated_image, boxes = detect_objects(rgb_image, self.target)
            self.X_target, self.Y_target, self.Z_target = find_xyz(self.env, self.obs, boxes, to_be_detected='target')

            world_x = self.Y_target
            world_y = -self.X_target
            world_z = -self.Z_target

            target_pos = self.current_pos + np.array([world_x+x_tol,world_y,world_z+z_tol])
            self.obs = move_to_pose_absolute(self.env, self.current_pos, target_pos, self.gripper_state, kp=self.kp)
            self.current_pos = self.obs['robot0_eef_pos']
    
    def move_to_pose_delta(self, x, y, z):
        """
        Moves the end-effector by a certain delta in the x, y and z directions in the camera's frame.
        """
        self.obs = move_to_pose_delta(self.env, self.obs, x, y, z, kp=self.kp, gripper_state=self.gripper_state)

    def set_gripper_state(self, state, wait_steps=50):
        """
        Opens/closes the gripper.
        Arguments:
        - state : 1.0 for opening, -1.0 for closing
        - wait_steps : idle time for opening/closing gripper
        """
        if state == 1.0:
            print("Grasping...")
        else:
            print("Releasing...")

        self.gripper_state = state
        action = np.zeros(7)
        action[6] = self.gripper_state
        for i in range(wait_steps):
            self.obs, self.reward, self.done, self.info = self.env.step(action)
            self.env.render() 

    def close(self):
        """
        Ends the simulation.
        """
        print("Simulation finished.")
        self.env.close()
