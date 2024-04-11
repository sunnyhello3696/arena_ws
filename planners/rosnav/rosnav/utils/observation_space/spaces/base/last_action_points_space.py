from typing import Tuple

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
            _spaces = (
                spaces.Box(
                    low=self._min_linear_vel,
                    high=self._max_linear_vel,
                    shape=(1,),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=self._min_translational_vel,
                    high=self._max_translational_vel,
                    shape=(1,),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=self._min_angular_vel,
                    high=self._max_angular_vel,
                    shape=(1,),
                    dtype=np.float32,
                ),
            )
        else:
            _spaces = []
            for i in range(self._action_points_num):
                _spaces.append(
                    spaces.Box(low=0, high=self._max_dist, shape=(1,), dtype=np.float32),
                    spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                )
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
