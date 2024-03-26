#! /usr/bin/env python3
import random
import re
import time
from typing import Tuple

import gymnasium
import numpy as np
import rospy
from flatland_msgs.msg import StepWorld
from geometry_msgs.msg import Twist
from rl_utils.utils.observation_collector.constants import DONE_REASONS
from rl_utils.utils.observation_collector.observation_manager import ObservationManager
from rl_utils.utils.observation_collector.observation_units.base_collector_unit import (
    BaseCollectorUnit,
)
from rl_utils.utils.observation_collector.observation_units.globalplan_collector_unit import (
    GlobalplanCollectorUnit,
)
from rl_utils.utils.observation_collector.observation_units.semantic_ped_unit import (
    SemanticAggregateUnit,
)
from rl_utils.utils.rewards.reward_function import RewardFunction
from rosnav.model.base_agent import BaseAgent
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from std_srvs.srv import Empty
from task_generator.shared import Namespace
from task_generator.task_generator_node import TaskGenerator
from task_generator.utils import rosparam_get

# convex_d86
from rl_utils.utils.observation_collector.observation_units.convex_collector_unit import (
    ConvexCollectorUnit,
)
from sensor_msgs.msg import Image  # Assuming conversion to Image message for rviz
# import ros_numpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock
import math
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS, TOPICS


def get_ns_idx(ns: str):
    try:
        return int(re.search(r"\d+", ns)[0])
    except Exception:
        return random.uniform(0, 3)
        # return 0.5


class FlatlandEnv(gymnasium.Env):
    """
    FlatlandEnv is an environment class that represents a Flatland environment for reinforcement learning.

    Args:
        ns (str): The namespace of the environment.
        agent_description (BaseAgent): The agent description.
        reward_fnc (str): The name of the reward function.
        max_steps_per_episode (int): The maximum number of steps per episode.
        trigger_init (bool): Whether to trigger the initialization of the environment.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        metadata (dict): The metadata of the environment.
        ns (str): The namespace of the environment.
        _agent_description (BaseAgent): The agent description.
        _debug_mode (bool): Whether the environment is in debug mode.
        _is_train_mode (bool): Whether the environment is in train mode.
        _step_size (float): The step size of the environment.
        _reward_fnc (str): The name of the reward function.
        _kwargs (dict): Additional keyword arguments.
        _steps_curr_episode (int): The current number of steps in the episode.
        _episode (int): The current episode number.
        _max_steps_per_episode (int): The maximum number of steps per episode.
        _last_action (np.ndarray): The last action taken in the environment.
        model_space_encoder (RosnavSpaceManager): The space encoder for the model.
        task (BaseTask): The task manager for the environment.
        reward_calculator (RewardFunction): The reward calculator for the environment.
        agent_action_pub (rospy.Publisher): The publisher for agent actions.
        _service_name_step (str): The name of the step world service.
        _step_world_publisher (rospy.Publisher): The publisher for the step world service.
        _step_world_srv (rospy.ServiceProxy): The service proxy for the step world service.
        observation_collector (ObservationManager): The observation collector for the environment.

    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ns: str,
        agent_description: BaseAgent,
        reward_fnc: str,
        max_steps_per_episode=100,
        trigger_init: bool = False,
        *args,
        **kwargs,
    ):
        super(FlatlandEnv, self).__init__()

        # /sim_{rank + 1}/sim_{rank + 1}_{robot_model}
        self.ns = Namespace(ns)
        print(f"FlatlandEnv: {self.ns}")

        # _agent_description: config["rl_agent"]["architecture_name"]
        self._agent_description = agent_description

        self._debug_mode = rospy.get_param("/debug_mode", False)

        if not self._debug_mode:
            rospy.init_node(f"env_{self.ns.simulation_ns}".replace("/", "_"))

        self._is_train_mode = rospy.get_param_cached("/train_mode", default=True)
        self._step_size = rospy.get_param_cached("/step_size")

        self._reward_fnc = reward_fnc
        self._kwargs = kwargs

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self._last_action = np.array([0, 0, 0])  # linear x, linear y, angular z

        self.enable_rviz = rospy.get_param("/if_viz", False)
        self.clock_sub = rospy.Subscriber(self.ns.oldname("clock"), Clock, self.clock_cb)
        self.clock_time = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 如果配置中的debug_mode=False，则trigger_init=True；debug_mode=True，不启用（False）。
        if not trigger_init:
            self.init()

    def init(self):
        """
        Initializes the environment.

        Returns:
            bool: True if the initialization is successful, False otherwise.
        """
        self.model_space_encoder = RosnavSpaceManager(
            space_encoder_class=self._agent_description.space_encoder_class,
            observation_spaces=self._agent_description.observation_spaces,
            observation_space_kwargs=self._agent_description.observation_space_kwargs,
        )

        if self._is_train_mode:
            self._setup_env_for_training(self._reward_fnc, **self._kwargs)

        # observation collector
        self.observation_collector = ObservationManager(
            ns=self.ns,
            obs_structur=[
                # BaseCollectorUnit,
                ConvexCollectorUnit,
                GlobalplanCollectorUnit,
                SemanticAggregateUnit,
            ],
        )
        return True

    @property
    def action_space(self):
        return self.model_space_encoder.get_action_space()

    @property
    def observation_space(self):
        return self.model_space_encoder.get_observation_space()

    def _setup_env_for_training(self, reward_fnc: str, **kwargs):
        # instantiate task manager
        task_generator = TaskGenerator(self.ns.simulation_ns)
        self.task = task_generator._get_predefined_task(**kwargs)

        # reward calculator
        self.reward_calculator = RewardFunction(
            rew_func_name=reward_fnc,
            holonomic=self.model_space_encoder._is_holonomic,
            robot_radius=self.task.robot_managers[0]._robot_radius, # 由training_config.yaml传入
            safe_dist=self.task.robot_managers[0].safe_distance, 
            goal_radius=rosparam_get(float, "goal_radius", 0.3),
        )

        self.agent_action_pub = rospy.Publisher(self.ns("cmd_vel"), Twist, queue_size=1)

        # service clients
        self._service_name_step = self.ns.simulation_ns("step_world")
        self._step_world_publisher = rospy.Publisher(
            self._service_name_step, StepWorld, queue_size=10
        )
        self._step_world_srv = rospy.ServiceProxy(
            self._service_name_step, Empty, persistent=True
        )

    def _pub_action(self, action: np.ndarray) -> Twist:
        assert len(action) == 3

        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]

        self.agent_action_pub.publish(action_msg)

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        return self.model_space_encoder.decode_action(action)

    def _encode_observation(self, observation, *args, **kwargs):

        if self.enable_rviz:
            
            _robot_pose = observation[OBS_DICT_KEYS.ROBOT_POSE]  # observation_space_manager.py
            # 创建 TransformStamped 消息
            t = TransformStamped()
            # 填充消息
            t.header.stamp = self.clock_time
            t.header.frame_id = "map"
            t.child_frame_id = self.ns.oldname("robot")
            t.transform.translation.x = _robot_pose.x
            t.transform.translation.y = _robot_pose.y
            t.transform.translation.z = 0.0  # 因为是2D，所以z坐标为0
            # 计算四元数（仅围绕Z轴旋转）
            cy = math.cos(_robot_pose.theta * 0.5)
            sy = math.sin(_robot_pose.theta * 0.5)
            cr = 1  # 绕X轴旋转的余弦值为1（没有旋转）
            sr = 0  # 绕X轴旋转的正弦值为0（没有旋转）
            cp = 1  # 绕Y轴旋转的余弦值为1（没有旋转）
            sp = 0  # 绕Y轴旋转的正弦值为0（没有旋转）
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            # 设置四元数
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            # 发布 tf
            self.tf_broadcaster.sendTransform(t)


        observation = self.model_space_encoder.encode_observation(observation, **kwargs)
        
        return observation

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple: A tuple containing the encoded observation, reward, done flag, info dictionary, and False flag.

        """

        decoded_action = self._decode_action(action)
        self._pub_action(decoded_action)

        if self._is_train_mode:
            self.call_service_takeSimStep()

        obs_dict = self.observation_collector.get_observations(
            last_action=self._last_action
        )
        self._last_action = decoded_action

        # calculate reward
        reward, reward_info = self.reward_calculator.get_reward(
            action=decoded_action,
            **obs_dict,
        )

        self._steps_curr_episode += 1

        # info
        info, done = self._determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )

        return (
            self._encode_observation(obs_dict, is_done=done),
            reward,
            done,
            False,
            info,
        )

    def call_service_takeSimStep(self, t: float = None, srv_call: bool = True):
        if srv_call:
            self._step_world_srv()
        request = StepWorld()
        request.required_time = self._step_size if t is None else t

        self._step_world_publisher.publish(request)

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: The random seed for the environment.
            options: Additional options for resetting the environment.

        Returns:
            tuple: A tuple containing the encoded observation and an empty info dictionary.

        """

        super().reset(seed=seed)
        self._episode += 1
        self.agent_action_pub.publish(Twist())

        first_map = self._episode <= 1 if "sim_1" in self.ns else False
        self.task.reset(
            first_map=first_map,
            reset_after_new_map=self._steps_curr_episode == 0,
        )
        self.reward_calculator.reset()
        self._steps_curr_episode = 0
        self._last_action = np.array([0, 0, 0])

        if self._is_train_mode:
            self.agent_action_pub.publish(Twist())
            self.call_service_takeSimStep(t=0.1)

        obs_dict = self.observation_collector.get_observations()
        info_dict = {}
        return (
            self._encode_observation(obs_dict),
            info_dict,
        )

    def close(self):
        """
        Close the environment.

        """
        pass

    def _determine_termination(
        self,
        reward_info: dict,
        curr_steps: int,
        max_steps: int,
        info: dict = None,
    ) -> Tuple[dict, bool]:
        """
        Determine if the episode should terminate.

        Args:
            reward_info (dict): The reward information.
            curr_steps (int): The current number of steps in the episode.
            max_steps (int): The maximum number of steps per episode.
            info (dict): Additional information.

        Returns:
            tuple: A tuple containing the info dictionary and a boolean flag indicating if the episode should terminate.

        """

        if info is None:
            info = {}

        terminated = reward_info["is_done"]

        if terminated:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]
            info["episode_length"] = self._steps_curr_episode

        if curr_steps >= max_steps:
            terminated = True
            info["done_reason"] = DONE_REASONS.STEP_LIMIT.name
            info["is_success"] = 0
            info["episode_length"] = self._steps_curr_episode

        return info, terminated
    
    def clock_cb(self,clock_msg: Clock):
        self.clock_time = clock_msg.clock