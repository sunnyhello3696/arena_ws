import numpy as np

from gymnasium import spaces

import rospy


class ActionSpaceManager:
    """
    Class representing an action space manager.

    Args:
        holonomic (bool): Flag indicating whether the robot is holonomic.
        action_space_discrete (bool): Flag indicating whether the action space is discrete.
        actions (dict): Dictionary containing the available actions.
        stacked (bool): Flag indicating whether the actions are stacked.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _holonomic (bool): Flag indicating whether the robot is holonomic.
        _discrete (bool): Flag indicating whether the action space is discrete.
        _actions (dict): Dictionary containing the available actions.
        _stacked (bool): Flag indicating whether the actions are stacked.
        _space: The action space.

    Properties:
        actions: Get the available actions.
        action_space: Get the action space.
        shape: Get the shape of the action space.
    """

    def __init__(
        self,
        holonomic: bool,
        action_space_discrete: bool,
        actions: dict,
        stacked: bool,
        normalize_points: bool,
        action_points_num: int,
        *args,
        **kwargs,
    ) -> None:
        self._holonomic = holonomic
        self._discrete = action_space_discrete
        self._actions = actions
        self._stacked = stacked
        self._normalize_points = normalize_points
        self._action_points_num = action_points_num*2
        self._action_dis_min = rospy.get_param_cached("action_dis_min", 0.0)

        self._space = self.get_action_space()

    @property
    def actions(self):
        """
        Get the available actions.

        Returns:
            dict: Dictionary containing the available actions.
        """
        return self._actions

    @property
    def action_space(self):
        """
        Get the action space.

        Returns:
            object: The action space.
        """
        return self._space

    @property
    def shape(self):
        """
        Get the shape of the action space.

        Returns:
            tuple: The shape of the action space.
        """
        return self._space.shape

    def get_action_space(self):
        """
        Get the action space based on the configuration.

        Returns:
            object: The action space object.
        """
        if not self._normalize_points:
            if self._discrete:
                return spaces.Discrete(len(self._actions))

            linear_range = self._actions["linear_range"]
            angular_range = self._actions["angular_range"]

            if not self._holonomic:
                return spaces.Box(
                    low=np.array([linear_range[0], angular_range[0]]),
                    high=np.array([linear_range[1], angular_range[1]]),
                    dtype=np.float32,
                )

            linear_range_x, linear_range_y = (
                linear_range["x"],
                linear_range["y"],
            )

            return spaces.Box(
                low=np.array(
                    [
                        linear_range_x[0],
                        linear_range_y[0],
                        angular_range[0],
                    ]
                ),
                high=np.array(
                    [
                        linear_range_x[1],
                        linear_range_y[1],
                        angular_range[1],
                    ]
                ),
                dtype=np.float32,
            )
        else:
            low_limit = np.array([0 if i % 2 == 0 else self._action_dis_min for i in range(self._action_points_num)])
            return spaces.Box(
                low=low_limit,
                high=np.array([1] * self._action_points_num),
                dtype=np.float32,
            )

    def decode_action(self, action):
        """
        Decode the action.

        Args:
            action: The action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        if not self._normalize_points:
            if type(action) == int:
                action = [action]

            if self._stacked:
                action = action[0] if action.ndim == 2 else action

            if self._discrete:
                return self._extend_action_array(self._translate_disc_action(action))

            return self._extend_action_array(action)
        else:
            if self._stacked:
                action = action[0] if action.ndim == 2 else action
            return action

    def _extend_action_array(self, action: np.ndarray) -> np.ndarray:
        """
        Extend the action array.

        Args:
            action (np.ndarray): The action array.

        Returns:
            np.ndarray: The extended action array.
        """
        if self._holonomic:
            assert (
                self._holonomic and len(action) == 3
            ), "Robot is holonomic but action with only two freedoms of movement provided"

            return action
        else:
            assert (
                not self._holonomic and len(action) == 2
            ), "Robot is non-holonomic but action with more than two freedoms of movement provided"
            return np.array([action[0], 0, action[1]])

    def _translate_disc_action(self, action: int):
        """
        Translate the discrete action.

        Args:
            action (int): The discrete action.

        Returns:
            np.ndarray: The translated action.
        """
        return np.array(
            [self._actions[action]["linear"], self._actions[action]["angular"]]
        )
