# rlca-ros
RL navigation policy trained on and intended for multi-agent systems.


<!-- ## Build & Install
1. Activate poetry shell
```bash
cd arena-rosnav # navigate to the arena-rosnav directory
poetry shell
```
2. Install Python packages
```bash
pip install mpi4py
``` -->
# Usage
## This planner can be chosen using the local_planner argument like so:
```bash
roslaunch arena_bringup start_arena.launch local_planner:=rlca # Make sure that your virtual env/poetry is activated
```
## For more details regarding usage, please refer to our [documentation](https://arena-rosnav.readthedocs.io/en/latest/user_guides/usage/)

## Original work
For more information about RLCA, please refer to the original publication [paper](https://arxiv.org/abs/1709.10082)
