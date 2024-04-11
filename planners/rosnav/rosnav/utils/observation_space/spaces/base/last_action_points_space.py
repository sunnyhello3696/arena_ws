from typing import Tuple
import rospy

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from ...observation_space_factory import SpaceFactory
from ...utils import stack_spaces
from ..base_observation_space import BaseObservationSpace

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS


@SpaceFactory.register("last_action_points")
class LastActionPointsSpace(BaseObservationSpace):
    """
    Observation space representing the last action points taken by the agent.

    """

    def __init__(
        self,
        normalize_points: bool = False,
        action_points_num: int = 0,
        goal_max_dist: float = 50,
        *args,
        **kwargs
    ) -> None:
        self._max_dist = goal_max_dist
        self._normalize_points = normalize_points
        self._action_points_num = action_points_num
        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym spaces for the last action points space.

        Returns:
            A tuple of gym spaces representing the last action points space.
        """
        if not self._normalize_points:
            rospy.logerr("Non-normalized action points are not supported yet.")
            raise NotImplementedError("Non-normalized action points are not supported yet.")
        else:
            _spaces = []
            for i in range(self._action_points_num):
                # 分别为每个维度添加空间
                _spaces.append(spaces.Box(low=0, high=self._max_dist, shape=(1,), dtype=np.float32))
                _spaces.append(spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32))
        return stack_spaces(*_spaces)

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        """
        Encodes the observation by extracting the last action from the observation dictionary.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded observation representing the last action.
        """
        # print("observation[OBS_DICT_KEYS.LAST_ACTION] ndarray shape: ", observation[OBS_DICT_KEYS.LAST_ACTION].shape)
        return observation[OBS_DICT_KEYS.LAST_ACTION_POINTS]
