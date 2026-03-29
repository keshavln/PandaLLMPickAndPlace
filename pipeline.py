import tkinter as tk
from tkinter import simpledialog, messagebox
from llm import extract_object_and_target
from manipulator_env import PickAndPlaceTask

# User input
root = tk.Tk()
root.withdraw()
user_input = simpledialog.askstring("Prompt", "What task would you like to execute?")

# Extraction of object and target
prompt = extract_object_and_target(user_input)
messagebox.showinfo("Aim", f"Object: {prompt.split(',')[0]} \nTarget: {prompt.split(',')[1]}")
root.destroy()

# Initializing environment and manipulation task
manipulator = PickAndPlaceTask(prompt, 640, 480, kp=1.0, render_detections=False)

"""
Manipulation sequence:
1. Move end-effector to object
2. Close gripper
3. Raise object to small height
4. Move end-effector to target location
5. Release the object
"""

manipulator.move_to_object(x_tol=0.0,z_tol=0.075)
manipulator.move_to_object(x_tol=0.05,z_tol=0.075)
manipulator.set_gripper_state(1.0)
manipulator.move_to_pose_delta(0.0,0.0,-0.075)
manipulator.move_to_target(x_tol=0.05,z_tol=0.175)
manipulator.set_gripper_state(-1.0, wait_steps=200)

manipulator.close()