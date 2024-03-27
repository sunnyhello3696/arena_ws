import contextlib
import json
import os
import sys

import numpy as np
import rospkg
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS
import rospy
from rl_utils.utils.observation_collector.observation_manager import ObservationManager
from rosnav import *
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import PolicyType
from rosnav.model.custom_sb3_policy import *
from rosnav.rosnav_space_manager.rosnav_space_manager import RosnavSpaceManager
from rosnav.srv import GetAction, GetActionResponse
from rosnav.utils.constants import VALID_CONFIG_NAMES
from rosnav.utils.observation_space.spaces.base_observation_space import (
    BaseObservationSpace,
)
from rosnav.utils.utils import (
    load_json,
    load_vec_normalize,
    load_yaml,
    make_mock_env,
    wrap_vec_framestack,
)
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from tools.ros_param_distributor import (
    determine_space_encoder,
    populate_discrete_action_space,
    populate_laser_params,
)
from std_msgs.msg import Int16

from rl_utils.utils.observation_collector.observation_units.convex_collector_unit import (
    ConvexCollectorUnit,
)
from rl_utils.utils.observation_collector.observation_units.globalplan_collector_unit import (
    GlobalplanCollectorUnit,
)
from rl_utils.utils.observation_collector.observation_units.semantic_ped_unit import (
    SemanticAggregateUnit,
)
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock
import math
import argparse



sys.modules["rl_agent"] = sys.modules["rosnav"]
sys.modules["rl_utils.rl_utils.utils"] = sys.modules["rosnav.utils"]

from typing import Any, Dict, List

from task_generator.shared import Namespace


class RosnavNode:
    DEFAULT_DONES = np.array([[False]])
    DEFAULT_INFOS = [{}]
    DEFAULT_EPS_START = np.array([True])

    def __init__(self, ns: Namespace = ""):
        """
        Initialize the RosnavNode class.

        Args:
            ns (Namespace, optional): The namespace for the node. Defaults to "".
        """
        self.ns = Namespace(ns)
        self.enable_rviz = rospy.get_param("/if_viz", False)

        # Agent name and path
        self.agent_name = rospy.get_param("agent_name","burger") # burger
        self.model_name = "burger_AGENT_66_2024_03_26__20_19_25"
        self.agent_path = RosnavNode._get_model_path(self.model_name)
        print("agent_path: ", self.agent_path)

        assert os.path.isdir(
            self.agent_path
        ), f"Model cannot be found at {self.agent_path}"

        # Load hyperparams
        self._hyperparams = RosnavNode._load_hyperparams(self.agent_path)
        rospy.set_param("rl_agent", self._hyperparams["rl_agent"])

        self._setup_action_space(self._hyperparams)

        populate_laser_params(self._hyperparams)

        # Get Architecture Name and retrieve Observation spaces
        architecture_name = self._hyperparams["rl_agent"]["architecture_name"]
        agent: BaseAgent = AgentFactory.instantiate(architecture_name)
        observation_spaces: List[BaseObservationSpace] = agent.observation_spaces
        observation_spaces_kwargs = agent.observation_space_kwargs
        space_encoder_class = agent.space_encoder_class

        # Load observation normalization and frame stacking
        self._load_env_wrappers(self._hyperparams, agent)

        # Set RosnavSpaceEncoder as Middleware
        self._encoder = RosnavSpaceManager(
            space_encoder_class=space_encoder_class,
            observation_spaces=observation_spaces,
            observation_space_kwargs=observation_spaces_kwargs,
            action_space_kwargs=None,
        )

        # Load the model
        self._agent = self._get_model(
            architecture_name=architecture_name,
            checkpoint_name=self._hyperparams["rl_agent"]["checkpoint"],
            agent_path=self.agent_path,
        )

        # self._observation_manager = ObservationManager(self.ns)
        self._observation_manager = ObservationManager(
            ns=self.ns,
            obs_structur=[
                # BaseCollectorUnit,
                ConvexCollectorUnit,
                GlobalplanCollectorUnit,
                SemanticAggregateUnit,
            ],
        )


        print("service name: ", self.ns("rosnav/get_action"))
        self._get_next_action_srv = rospy.Service(
            self.ns("rosnav/get_action"), GetAction, self._handle_next_action_srv
        )
        self._sub_reset_stacked_obs = rospy.Subscriber(
            "/scenario_reset", Int16, self._on_scene_reset
        )

        self.state = None
        self._last_action = [0, 0, 0]
        self._reset_state = True
        self.clock_sub = rospy.Subscriber(self.ns("clock"), Clock, self.clock_cb)
        self.clock_time = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        print("RosnavNode initialized")

    def _setup_action_space(self, hyperparams: dict):
        is_action_space_discrete = (
            hyperparams["rl_agent"]["discrete_action_space"]
            if "discrete_action_space" in self._hyperparams["rl_agent"]
            else self._hyperparams["rl_agent"]["action_space"]["discrete"]
        )
        rospy.set_param("rl_agent/action_space/discrete", is_action_space_discrete)

        if is_action_space_discrete:
            populate_discrete_action_space(hyperparams)

    def _load_env_wrappers(self, hyperparams: dict, agent_description: BaseAgent):
        """
        Loads the environment wrappers based on the provided hyperparameters and agent description.

        Args:
            hyperparams (dict): The hyperparameters for the RL agent.
            agent_description (BaseAgent): The description of the agent.

        Returns:
            None
        """
        # Load observation normalization and frame stacking
        self._normalized_mode = hyperparams["rl_agent"]["normalize"]
        self._reduced_laser_mode = (
            hyperparams["rl_agent"]["laser"]["reduce_num_beams"]["enabled"]
            if "laser" in hyperparams["rl_agent"]
            else False
        )
        self._stacked_mode = (
            hyperparams["rl_agent"]["frame_stacking"]["enabled"]
            if "frame_stacking" in hyperparams["rl_agent"]
            else False
        )

        if self._stacked_mode:
            self._vec_stacked = RosnavNode._get_vec_stacked(
                agent_description, self._hyperparams
            )
            self._stacked_obs_container = self._vec_stacked.stacked_obs
        else:
            self._vec_stacked = None

        if self._normalized_mode:
            self._vec_normalize = RosnavNode._get_vec_normalize(
                agent_description, self.agent_path, self._hyperparams, self._vec_stacked
            )

    def _encode_observation(self, observation: Dict[str, Any]):
        """
        Encodes the given observation using the encoder.

        Args:
            observation (Dict[str, Any]): The observation to be encoded.

        Returns:
            The encoded observation.
        """
        if self.enable_rviz and self.clock_time is not None:
            
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
        return self._encoder.encode_observation(observation)

    def _get_observation(self):
        """
        Get the observation from the observation manager and append the last action.

        Returns:
            dict: The observation dictionary.
        """
        observation = self._observation_manager.get_observations()
        observation[OBS_DICT_KEYS.LAST_ACTION] = self._last_action
        return observation

    def get_action(self):
        """
        Get the action to be taken based on the current observation.

        Returns:
            The decoded action to be taken.
        """
        observation = self._encode_observation(self._get_observation())

        if self._stacked_mode:
            observation, _ = self._stacked_obs_container.update(
                observation, RosnavNode.DEFAULT_DONES, RosnavNode.DEFAULT_INFOS
            )

        if self._normalized_mode:
            try:
                observation = self._vec_normalize.normalize_obs(observation)
            except ValueError as e:
                rospy.logerr(e)
                rospy.logerr(
                    "Check if the configuration file correctly specifies the observation space."
                )
                rospy.signal_shutdown("")

        predict_dict = {"observation": observation, "deterministic": True}

        if self._recurrent_arch:
            predict_dict.update(
                {
                    "state": self.state,
                    "episode_start": (
                        RosnavNode.DEFAULT_EPS_START if self._reset_state else None
                    ),
                }
            )
            self._reset_state = False

        action, self.state = self._agent.predict(**predict_dict)

        decoded_action = self._encoder.decode_action(action)
        
        print("=======================")
        print("action: ", action)
        print("decoded_action: ", decoded_action)

        self._last_action = decoded_action

        return decoded_action

    def _handle_next_action_srv(self, request: GetAction):
        """
        Handles the service request to get the next action.

        Args:
            request (GetAction): The service request.

        Returns:
            GetActionResponse: The service response containing the next action.
        """
        action = self.get_action()

        response = GetActionResponse()
        response.action = action

        return response

    def _on_scene_reset(self, request: Int16):
        """
        Resets the last action and stacked observations.

        Args:
            request (Int16): The reset request.

        Returns:
            None
        """
        self._reset_last_action()
        self._reset_stacked_obs()

    def _reset_last_action(self):
        """
        Resets the last action to [0, 0, 0].
        """
        self._last_action = [0, 0, 0]

    def _reset_stacked_obs(self):
        """
        Resets the stacked observation.

        This method sets the `_reset_state` flag to True, clears the `state` variable,
        and resets the stacked observation container if the stacked mode is enabled.
        """
        self._reset_state = True
        self.state = None

        if self._stacked_mode:
            observation = self._encode_observation(self._get_observation())
            self._stacked_obs_container.reset(observation)

    def _get_model(self, architecture_name: str, checkpoint_name: str, agent_path: str):
        """
        Get the model based on the given architecture name, checkpoint name, and agent path.

        Args:
            architecture_name (str): The name of the architecture.
            checkpoint_name (str): The name of the checkpoint.
            agent_path (str): The path to the agent.

        Returns:
            policy: The loaded policy model.
        """
        net_type: PolicyType = AgentFactory.registry[architecture_name].type
        model_path = os.path.join(agent_path, f"{checkpoint_name}.zip")

        if not net_type or net_type != PolicyType.MLP_LSTM:
            self._recurrent_arch = False
            return PPO.load(model_path).policy
        else:
            self._recurrent_arch = True
            return RecurrentPPO.load(model_path).policy
        
    def clock_cb(self,clock_msg: Clock):
        self.clock_time = clock_msg.clock

    @staticmethod
    def _get_model_path(model_name):
        return os.path.join(rospkg.RosPack().get_path("rosnav"), "agents", model_name)

    @staticmethod
    def _load_hyperparams(agent_path):
        for cfg_name in VALID_CONFIG_NAMES:
            cfg_path = os.path.join(agent_path, cfg_name)
            if os.path.isfile(cfg_path):
                if cfg_name.endswith(".json"):
                    cfg_dict = {"rl_agent": load_json(cfg_path)}
                elif cfg_name.endswith(".yaml"):
                    cfg_dict = load_yaml(cfg_path)

                assert cfg_dict is not None, "Config file is empty."
                return cfg_dict
        raise ValueError("No valid config file found in agent folder.")

    @staticmethod
    def _get_vec_normalize(
        agent_description: BaseAgent, agent_path: str, hyperparams: dict, venv=None
    ):
        """
        Get the vector normalizer for the RL agent.

        Args:
            agent_description (str): Description of the agent.
            agent_path (str): Path to the agent.
            hyperparams (dict): Hyperparameters for the RL agent.
            venv (object, optional): Virtual environment. Defaults to None.

        Returns:
            object: Vector normalizer for the RL agent.
        """
        if venv is None:
            venv = make_mock_env(agent_description)
        checkpoint = hyperparams["rl_agent"]["checkpoint"]
        vec_normalize_path = os.path.join(agent_path, f"vec_normalize_{checkpoint}.pkl")
        return load_vec_normalize(vec_normalize_path, hyperparams["rl_agent"], venv)

    @staticmethod
    def _get_vec_stacked(agent_description: BaseAgent, hyperparams: dict):
        """
        Returns a vectorized environment with frame stacking.

        Args:
            agent_description (str): Description of the agent.
            hyperparams (dict): Hyperparameters for the RL agent.

        Returns:
            Vectorized environment with frame stacking.
        """
        venv = make_mock_env(agent_description)
        return wrap_vec_framestack(
            venv, hyperparams["rl_agent"]["frame_stacking"]["stack_size"]
        )
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ns", "--namespace", type=str)

    return parser.parse_known_args()[0]

if __name__ == "__main__":
    rospy.init_node("rosnav_node")
    print("rosnav_node started")
    args = parse_args()
    node = RosnavNode(ns=args.namespace)

    while not rospy.is_shutdown():
        rospy.spin()
