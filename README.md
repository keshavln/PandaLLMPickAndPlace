# Panda Pick and Place with Grounding DINO and Llama 3.3 70b Versatile
## Overview
This repository contains the source code of a MuJoCo simulation pipeline that allows the Panda robotic arm to perform pick and place tasks based on natural language inputs. This system uses:
- Llama 3.3 70B Versatile to extract the object and target destination from a user text input
- Grounding DINO for zero-shot object detection. Grounding DINO was chosen due to its ease of implementation and high accuracy in the MuJoCo environment.
- Robosuite and MuJoCo for manipulation.

The MuJoCo scene contains:
- A red cube
- A blue cube
- A cross

## Demo



https://github.com/user-attachments/assets/ac719464-bccc-4db3-92bc-6a00be245674



## Usage
```
# Clone this repository
git clone https://github.com/keshavln/PandaLLMPickAndPlace.git
cd PandaLLMPickAndPlace

# Install requirements
pip install -r requirements.txt
```
Additionally, follow the instructions in the [this Colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb#scrollTo=Nkrkkj7CUlkh) to install Grounding DINO:

## Detailed Look

Execution sequence:
1. User prompt is given
2. Object and target are extracted by the LLM and passed to Grounding DINO
3. Based on the bounding box coordinates and depth data, the approximate 3D coordinates of the object and target in the camera’s frame are calculated using the pinhole camera model. 

   ```
   Z = depth[v, u]
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   ```

   These are then transformed to the MuJoCo world’s frame before they are passed to the controller. Since the camera aligns with the global z axis, this transformation is relatively straightforward.

   ```
   world_x = camera_y
   world_y = -camera_x
   world_z = -camera_z
   ```
4. The end-effector moves to the calculated object coordinates, grasps the object, lifts it vertically by a customizable distance, moves to the target’s coordinates, and releases the object.

Parallax error poses a challenge by introducing difficulty in aligning the gripper precisely with the object. If the object was not initially positioned directly underneath the camera and instead at an angle, the end-effector would not accurately position itself above the object. It is for this reason that a second iteration is run. Once the end-effector finishes moving to the approximate location of the object, object detection and coordinate transformation are run once more, producing an updated, more accurate set of coordinates to which the end-effector then proceeds. Any such number of iterations can be applied but two proves to be more than enough.

Files used:
Files used:
- pipeline.py - entry point
- llm.py - contains functions relevant to llm inference using the Groq API
- perception.py - contains functions that perform zero shot object detection using Grounding DINO
- projection.py - contains a function to calculate 3D coordinates from bounding box center pixel coordinates.
- robot_control.py - contains functions to move end-effector to relative and absolute positions
- manipulator_env.py - defines mujoco simulation environment and defines a class for a pick and place manipulation task which includes functions to detect and approach both the object and the target.

## Limitations and Future Scope

- As of now, the end-effector does not undergo any sort of rotation. This would make it difficult to grip a cuboidal object that is placed at an angle.
- The x and z tolerances need to be calibrated manually.
- Considering multiple different camera angles could result in more accurate object and target coordinates and could eliminate the need for two iterations to achieve end-effector alignment.
