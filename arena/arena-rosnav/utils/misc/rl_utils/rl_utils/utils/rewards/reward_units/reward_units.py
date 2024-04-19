import random
from typing import Any, Callable, Dict
from warnings import warn

import numpy as np
import rospy
from rl_utils.utils.observation_collector.constants import DONE_REASONS, OBS_DICT_KEYS

from ..constants import DEFAULTS, REWARD_CONSTANTS
from ..reward_function import RewardFunction
from ..utils import check_params, get_ped_type_min_distances
from .base_reward_units import GlobalplanRewardUnit, RewardUnit
from .reward_unit_factory import RewardUnitFactory

# UPDATE WHEN ADDING A NEW UNIT
__all__ = [
    "RewardGoalReached",
    "RewardSafeDistance",
    "RewardNoMovement",
    "RewardApproachGoal",
    "RewardCollision",
    "RewardDistanceTravelled",
    "RewardApproachGlobalplan",
    "RewardFollowGlobalplan",
    "RewardReverseDrive",
    "RewardAbruptVelocityChange",
    "RewardRootVelocityDifference",
    "RewardTwoFactorVelocityDifference",
    "RewardActiveHeadingDirection",
    "RewardSafeDistanceExp",
    "RewardFixedStep",
    "RewardFollowTebplan",
    "RewardActionPointsChange",
]

if_show_reward = False


@RewardUnitFactory.register("goal_reached")
class RewardGoalReached(RewardUnit):
    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.SUCCESS.name,
        "is_success": True,
    }
    NOT_DONE_INFO = {"is_done": False}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.GOAL_REACHED.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when the goal is reached.

        Args:
            reward_function (RewardFunction): The reward function object holding this unit.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.GOAL_REACHED.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.GOAL_REACHED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward
        self._goal_radius = self._reward_function.goal_radius

    def check_parameters(self, *args, **kwargs):
        if self._reward < 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Negative rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, distance_to_goal: float, *args: Any, **kwargs: Any) -> None:
        """Calculates the reward and updates the information when the goal is reached.

        Args:
            distance_to_goal (float): Distance to the goal in m.
        """
        if distance_to_goal < self._reward_function.goal_radius:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)
            if if_show_reward:
                self._sum_reward += self._reward
        else:
            self.add_info(self.NOT_DONE_INFO)

    def reset(self):
        self._goal_radius = self._reward_function.goal_radius
        if if_show_reward:
            print("GoalReached reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("safe_distance")
class RewardSafeDistance(RewardUnit):
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.SAFE_DISTANCE.REWARD,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when violating the safe distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to DEFAULTS.SAFE_DISTANCE.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward
        self._safe_dist = self._reward_function._safe_dist

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, *args: Any, **kwargs: Any):
        violation_in_blind_spot = False
        if "full_laser_scan" in kwargs and len(kwargs["full_laser_scan"]) > 0:
            violation_in_blind_spot = kwargs["full_laser_scan"].min() <= self._safe_dist

        if (
            self.get_internal_state_info("safe_dist_breached")
            or violation_in_blind_spot
        ):
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)

            if if_show_reward:
                self._sum_reward += self._reward
                
    def reset(self):
        if if_show_reward:
            print("SafeDistance reward:", self._sum_reward)
            self._sum_reward = 0

@RewardUnitFactory.register("safe_distance_exp")
class RewardSafeDistanceExp(RewardUnit):
    SAFE_DIST_VIOLATION_INFO = {"safe_dist_violation": True}

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.SAFE_DISTANCE.REWARD,
        w_factor: float = 1.0,
        punish_dist: float = 1.0,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when violating the safe distance.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for violating the safe distance. Defaults to DEFAULTS.SAFE_DISTANCE.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward
        self._safe_dist = self._reward_function._safe_dist
        self._w_factor = w_factor
        self._punish_dist = punish_dist
        # print("self._w_factor",self._w_factor)

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, *args: Any, **kwargs: Any):
        violation_in_blind_spot = False
        laser_min = self.get_internal_state_info("min_dist_laser")
        if "full_laser_scan" in kwargs and len(kwargs["full_laser_scan"]) > 0:
            laser_min = kwargs["full_laser_scan"].min()
        if laser_min <= self._punish_dist:
            self._reward = self._reward * np.exp(-1.0*self._w_factor * (laser_min-0.))
            self.add_reward(self._reward)
            self.add_info(self.SAFE_DIST_VIOLATION_INFO)
            if if_show_reward:
                self._sum_reward += self._reward
    
    def reset(self):
        if if_show_reward:
            print("SafeDistanceExp reward:", self._sum_reward)
            self._sum_reward = 0

@RewardUnitFactory.register("fixed_per_step")
class RewardFixedStep(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.NO_MOVEMENT.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """
        Class for calculating the reward of a fixed value per step.


        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any):
            self.add_reward(self._reward)
            if if_show_reward:
                self._sum_reward += self._reward
    
    def reset(self):
        if if_show_reward:
            print("FixedStep reward:", self._sum_reward)
            self._sum_reward = 0

@RewardUnitFactory.register("no_movement")
class RewardNoMovement(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.NO_MOVEMENT.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when there is no movement.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for no movement. Defaults to DEFAULTS.NO_MOVEMENT.REWARD.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.NO_MOVEMENT._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any):
        if (
            action is not None
            and abs(action[0]) <= REWARD_CONSTANTS.NO_MOVEMENT_TOLERANCE
        ):
            self.add_reward(self._reward)
            if if_show_reward:
                self._sum_reward += self._reward

    def reset(self):    
        if if_show_reward:
            print("NoMovement reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("approach_goal")
class RewardApproachGoal(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GOAL.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GOAL.NEG_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when approaching the goal.

        Args:
            reward_function (RewardFunction): The reward function object.
            pos_factor (float, optional): Positive factor for approaching the goal. Defaults to DEFAULTS.APPROACH_GOAL.POS_FACTOR.
            neg_factor (float, optional): Negative factor for distancing from the goal. Defaults to DEFAULTS.APPROACH_GOAL.NEG_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.APPROACH_GOAL._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._pos_factor = pos_factor
        self._neg_factor = neg_factor
        self.last_goal_dist = None

    def check_parameters(self, *args, **kwargs):
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"[{self.__class__.__name__}] Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)

    def __call__(self, distance_to_goal, *args, **kwargs):
        if self.last_goal_dist is not None:
            w = (
                self._pos_factor
                if (self.last_goal_dist - distance_to_goal) > 0
                else self._neg_factor
            )
            self.add_reward(w * (self.last_goal_dist - distance_to_goal))
            if if_show_reward:
                self._sum_reward += w * (self.last_goal_dist - distance_to_goal)
        self.last_goal_dist = distance_to_goal

    def reset(self):
        self.last_goal_dist = None
        if if_show_reward:
            print("ApproachGoal reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("collision")
class RewardCollision(RewardUnit):
    DONE_INFO = {
        "is_done": True,
        "done_reason": DONE_REASONS.COLLISION.name,
        "is_success": False,
    }

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.COLLISION.REWARD,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward when a collision is detected.

        Args:
            reward_function (RewardFunction): The reward function object.
            reward (float, optional): The reward value for reaching the goal. Defaults to DEFAULTS.COLLISION.REWARD.
        """
        super().__init__(reward_function, True, *args, **kwargs)
        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        coll_in_blind_spots = False
        if "full_laser_scan" in kwargs:
            if len(kwargs["full_laser_scan"]) > 0:
                coll_in_blind_spots = (
                    kwargs["full_laser_scan"].min() <= self.robot_radius
                )

        laser_min = self.get_internal_state_info("min_dist_laser")
        if laser_min <= self.robot_radius or coll_in_blind_spots:
            self.add_reward(self._reward)
            self.add_info(self.DONE_INFO)
            if if_show_reward:
                self._sum_reward += self._reward
    
    def reset(self):
        if if_show_reward:
            print("Collision reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("distance_travelled")
class RewardDistanceTravelled(RewardUnit):
    def __init__(
        self,
        reward_function: RewardFunction,
        consumption_factor: float = DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR,
        lin_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR,
        ang_vel_scalar: float = DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR,
        _on_safe_dist_violation: bool = DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward for the distance travelled.

        Args:
            reward_function (RewardFunction): The reward function object.
            consumption_factor (float, optional): Negative consumption factor. Defaults to DEFAULTS.DISTANCE_TRAVELLED.CONSUMPTION_FACTOR.
            lin_vel_scalar (float, optional): Scalar for the linear velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.LIN_VEL_SCALAR.
            ang_vel_scalar (float, optional): Scalar for the angular velocity. Defaults to DEFAULTS.DISTANCE_TRAVELLED.ANG_VEL_SCALAR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.DISTANCE_TRAVELLED._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._factor = consumption_factor
        self._lin_vel_scalar = lin_vel_scalar
        self._ang_vel_scalar = ang_vel_scalar

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if action is None:
            return
        lin_vel, ang_vel = action[0], action[-1]
        reward = (
            (lin_vel * self._lin_vel_scalar) + (ang_vel * self._ang_vel_scalar)
        ) * -self._factor
        self.add_reward(reward)
        if if_show_reward:
            self._sum_reward += reward

    def reset(self):
        if if_show_reward:
            print("DistanceTravelled reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("approach_globalplan")
class RewardApproachGlobalplan(GlobalplanRewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        pos_factor: float = DEFAULTS.APPROACH_GLOBALPLAN.POS_FACTOR,
        neg_factor: float = DEFAULTS.APPROACH_GLOBALPLAN.NEG_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.APPROACH_GLOBALPLAN._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        """Class for calculating the reward for approaching the global plan.

        Args:
            reward_function (RewardFunction): The reward function object.
            pos_factor (float, optional): Positive factor for approaching the goal. Defaults to DEFAULTS.APPROACH_GLOBALPLAN.POS_FACTOR.
            neg_factor (float, optional): Negative factor for distancing from the goal. Defaults to DEFAULTS.APPROACH_GLOBALPLAN.NEG_FACTOR.
            _on_safe_dist_violation (bool, optional): Flag to indicate if there is a violation of safe distance. Defaults to DEFAULTS.APPROACH_GLOBALPLAN._ON_SAFE_DIST_VIOLATION.
        """
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._pos_factor = pos_factor
        self._neg_factor = neg_factor

        self.last_dist_to_path = None
        self._kdtree = None

    def check_parameters(self, *args, **kwargs):
        if self._pos_factor < 0 or self._neg_factor < 0:
            warn_msg = (
                f"[{self.__class__.__name__}] Both factors should be positive. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)
        if self._pos_factor >= self._neg_factor:
            warn_msg = (
                "'pos_factor' should be smaller than 'neg_factor' otherwise rotary trajectories will get rewarded. "
                f"Current values: [pos_factor={self._pos_factor}], [neg_factor={self._neg_factor}]"
            )
            warn(warn_msg)

    def __call__(
        self, global_plan: np.ndarray, robot_pose, *args: Any, **kwargs: Any
    ) -> Any:
        super().__call__(global_plan=global_plan, robot_pose=robot_pose)

        if self.curr_dist_to_path and self.last_dist_to_path:
            self.add_reward(self._calc_reward())

        self.last_dist_to_path = self.curr_dist_to_path

    def _calc_reward(self) -> float:
        w = (
            self._pos_factor
            if self.curr_dist_to_path < self.last_dist_to_path
            else self._neg_factor
        )
        return w * (self.last_dist_to_path - self.curr_dist_to_path)

    def reset(self):
        super().reset()
        self.last_dist_to_path = None


@RewardUnitFactory.register("follow_globalplan")
class RewardFollowGlobalplan(GlobalplanRewardUnit):
    """
    RewardFollowGlobalplan is a reward unit that calculates the reward based on the agent's
    distance to the global plan and its action.

    Args:
        reward_function (RewardFunction): The reward function to use for calculating the reward.
        min_dist_to_path (float, optional): The minimum distance to the global plan. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN.MIN_DIST_TO_PATH.
        reward_factor (float, optional): The reward factor to multiply the action by. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN.REWARD_FACTOR.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to handle safe distance violations. Defaults to DEFAULTS.FOLLOW_GLOBALPLAN._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _min_dist_to_path (float): The minimum distance to the global plan.
        _reward_factor (float): The reward factor to multiply the action by.

    Methods:
        __call__(self, action, global_plan, robot_pose, *args, **kwargs): Calculates the reward based on the agent's distance to the global plan and its action.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        min_dist_to_path: float = DEFAULTS.FOLLOW_GLOBALPLAN.MIN_DIST_TO_PATH,
        reward_factor: float = DEFAULTS.FOLLOW_GLOBALPLAN.REWARD_FACTOR,
        _on_safe_dist_violation: bool = DEFAULTS.FOLLOW_GLOBALPLAN._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._min_dist_to_path = min_dist_to_path
        self._reward_factor = reward_factor

    def __call__(
        self,
        action: np.ndarray,
        global_plan: np.ndarray,
        robot_pose,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Calculates the reward based on the given action, global plan, and robot pose.

        Args:
            action (np.ndarray): The action taken by the agent.
            global_plan (np.ndarray): The global plan for the robot.
            robot_pose: The current pose of the robot.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The calculated reward.
        """
        super().__call__(global_plan=global_plan, robot_pose=robot_pose)

        if (
            self.curr_dist_to_path
            and action is not None
            and self.curr_dist_to_path <= self._min_dist_to_path
        ):
            self.add_reward(self._reward_factor * action[0])


@RewardUnitFactory.register("reverse_drive")
class RewardReverseDrive(RewardUnit):
    """
    A reward unit that provides a reward for driving in reverse.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        reward (float, optional): The reward value for driving in reverse. Defaults to DEFAULTS.REVERSE_DRIVE.REWARD.
        _on_safe_dist_violation (bool, optional): Whether to penalize for violating safe distance. Defaults to DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _reward (float): The reward value for driving in reverse.

    Methods:
        check_parameters: Checks if the reward value is positive and issues a warning if it is.
        __call__: Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

    """

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        reward: float = DEFAULTS.REVERSE_DRIVE.REWARD,
        _on_safe_dist_violation: bool = DEFAULTS.REVERSE_DRIVE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._reward = reward

    def check_parameters(self, *args, **kwargs):
        """
        Checks if the reward value is positive and issues a warning if it is.
        """
        if self._reward > 0.0:
            warn_msg = (
                f"[{self.__class__.__name__}] Reconsider this reward. "
                f"Positive rewards may lead to unfavorable behaviors. "
                f"Current value: {self._reward}"
            )
            warn(warn_msg)

    def __call__(self, action: np.ndarray, *args, **kwargs):
        """
        Adds the reward value to the total reward if the action is not None and the first element of the action is less than 0.

        Args:
            action (np.ndarray): The action taken.

        """
        if action is not None and action[0] < 0:
            self.add_reward(self._reward)
            if if_show_reward:
                self._sum_reward += self._reward

    def reset(self):
        if if_show_reward:
            print("ReverseDrive reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("abrupt_velocity_change")
class RewardAbruptVelocityChange(RewardUnit):
    """
    A reward unit that penalizes abrupt changes in velocity.
    计算当前动作和上一动作在对应维度上的速度差异（vel_diff）。
    根据速度差异的四次方与因子的乘积计算奖励，使用-((vel_diff**4 / 100) * factor)公式。

    Args:
        reward_function (RewardFunction): The reward function to be used.
        vel_factors (Dict[str, float], optional): Velocity factors for each dimension. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize abrupt velocity changes on safe distance violation. Defaults to DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION.

    Attributes:
        _vel_factors (Dict[str, float]): Velocity factors for each dimension.
        last_action (np.ndarray): The last action taken.
        _vel_change_fcts (List[Callable[[np.ndarray], None]]): List of velocity change functions.

    Methods:
        _get_vel_change_fcts(): Returns a list of velocity change functions.
        _prepare_reward_function(idx: int, factor: float) -> Callable[[np.ndarray], None]: Prepares a reward function for a specific dimension.
        __call__(action: np.ndarray, *args, **kwargs): Calculates the reward based on the action taken.
        reset(): Resets the last action to None.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        vel_factors: Dict[str, float] = DEFAULTS.ABRUPT_VEL_CHANGE.VEL_FACTORS,
        _on_safe_dist_violation: bool = DEFAULTS.ABRUPT_VEL_CHANGE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._vel_factors = vel_factors
        self.last_action = None

        self._vel_change_fcts = self._get_vel_change_fcts()

    def _get_vel_change_fcts(self):
        return [
            self._prepare_reward_function(int(idx), factor)
            for idx, factor in self._vel_factors.items()
        ]

    # 计算给定动作action在指定维度上的速度变化，并根据变化量和因子计算奖励。
    def _prepare_reward_function(
        self, idx: int, factor: float
    ) -> Callable[[np.ndarray], None]:
        def vel_change_fct(action: np.ndarray):
            assert isinstance(self.last_action, np.ndarray)
            vel_diff = abs(action[idx] - self.last_action[idx])
            self.add_reward(-((vel_diff**4 / 100) * factor))
            if if_show_reward:
                self._sum_reward += -((vel_diff**4 / 100) * factor)

        return vel_change_fct

    def __call__(self, action: np.ndarray, *args, **kwargs):
        if self.last_action is not None:
            for rew_fct in self._vel_change_fcts:
                rew_fct(action)
        self.last_action = action

    def reset(self):
        self.last_action = None
        if if_show_reward:
            print("AbruptVelocityChange reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("root_velocity_difference")
class RewardRootVelocityDifference(RewardUnit):
    """
    A reward unit that calculates the difference in root velocity between consecutive actions.

    计算连续动作之间根速度的差异，并根据这种差异来计算奖励。
    这种方式鼓励算法生成动作序列，其中每个动作与前一个动作之间的速度差异最小化，从而促进平滑的运动转换。
    Args:
        reward_function (RewardFunction): The reward function to be used.
        k (float, optional): The scaling factor for the velocity difference. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize for violating safe distance.
            Defaults to DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _k (float): The scaling factor for the velocity difference.
        last_action (numpy.ndarray): The last action taken.

    Methods:
        __call__(self, action: np.ndarray, *args, **kwargs): Calculates the reward based on the velocity difference between
            the current action and the last action.
        reset(self): Resets the last action to None.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        k: float = DEFAULTS.ROOT_VEL_DIFF.K,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._k = k
        self.last_action = None

    def __call__(self, action: np.ndarray, *args, **kwargs):
        """
        Calculates and adds the reward based on the given action.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            None
        """
        if self.last_action is not None:
            # 计算当前动作与上一个动作之间的差异的平方的欧氏范数。实际上是计算二者之间速度差异的平方的根
            vel_diff = np.linalg.norm((action - self.last_action) ** 2)
            if vel_diff < self._k:
                self.add_reward((1 - vel_diff) / self._k)
        self.last_action = action

    def reset(self):
        self.last_action = None


@RewardUnitFactory.register("two_factor_velocity_difference")
class RewardTwoFactorVelocityDifference(RewardUnit):
    """
    A reward unit that calculates the difference in velocity between consecutive actions
    and penalizes the agent based on the squared difference.
    计算连续动作之间速度的变化，根据这些变化的平方差来计算奖励

    Args:
        reward_function (RewardFunction): The reward function to be used.
        alpha (float, optional): The weight for the squared difference in the first dimension of the action. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        beta (float, optional): The weight for the squared difference in the last dimension of the action. Defaults to DEFAULTS.ROOT_VEL_DIFF.K.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize the agent on safe distance violation. Defaults to DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        alpha: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.ALPHA,
        beta: float = DEFAULTS.TWO_FACTOR_VEL_DIFF.BETA,
        _on_safe_dist_violation: bool = DEFAULTS.ROOT_VEL_DIFF._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)

        self._alpha = alpha
        self._beta = beta
        self.last_action = None

    def __call__(self, action: np.ndarray, *args, **kwargs):
        """
        Calculates and adds the reward based on the difference between the current action and the last action.

        Args:
            action (np.ndarray): The current action.

        Returns:
            None
        """
        if self.last_action is not None:
            diff = (action - self.last_action) ** 2
            self.add_reward(-(diff[0] * self._alpha + diff[-1] * self._beta))
            if if_show_reward:
                self._sum_reward += -(diff[0] * self._alpha + diff[-1] * self._beta)
        self.last_action = action

    def reset(self):
        self.last_action = None
        if if_show_reward:
            print("TwoFactorVelocityDifference reward:", self._sum_reward)
            self._sum_reward = 0


@RewardUnitFactory.register("action_points_change")
class RewardActionPointsChange(RewardUnit):
    """
    A reward unit that penalizes changes in action points.
    Calculates the difference between consecutive action points in terms of Euclidean distance,
    and applies a linear penalty based on these differences.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        penalty_factor (float, optional): The factor that scales the linear penalty. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        last_action (numpy.ndarray): The last action taken in terms of action points.
        penalty_factor (float): The factor that scales the penalty.

    Methods:
        __call__(self, action: np.ndarray, *args, **kwargs): Calculates the reward based on the action points change.
        reset(self): Resets the last action to None.
    """

    @check_params
    def __init__(
        self,
        reward_function: RewardFunction,
        penalty_factor: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, *args, **kwargs)
        self.last_action_points = None
        self.penalty_factor = penalty_factor
        self.is_normalize_points = rospy.get_param_cached("is_normalize_points", False)
        self.action_points_num = rospy.get_param_cached("action_points_num", 0)

    def check_parameters(self, *args, **kwargs):
        """
        Checks if the reward value is positive and issues a warning if it is.
        """
        if not self.is_normalize_points:
            rospy.logerr("RewardActionPointsChange only use at Action points mode")
            raise ValueError("RewardActionPointsChange only use at Action points mode")
        
        if self.is_normalize_points and self.action_points_num == 0:
            rospy.logerr("Action points num is 0")
            raise ValueError("Action points num is 0")

    def __call__(self, action_points: np.ndarray, *args, **kwargs):
        """
        Calculates and adds the reward based on the given action points.

        Args:
            action (np.ndarray): The current action points.

        Returns:
            None
        """
        if action_points is None:
            rospy.logerr("RewardActionPointsChange Action points is None")
            raise ValueError("RewardActionPointsChange Action points is None")
        if self.last_action_points is not None:
            # Calculate the Euclidean distance between consecutive action points
            distances = np.linalg.norm(action_points - self.last_action_points, axis=1)
            # Apply linear penalty
            penalty = np.sum(distances) * self.penalty_factor
            self.add_reward(-penalty)
            if if_show_reward:
                self._sum_reward += -penalty
        self.last_action_points = action_points.copy()  # Store the current action for next comparison

    def reset(self):
        self.last_action_points = None
        if if_show_reward:
            print("ActionPointsChange reward:", self._sum_reward)
            print("====================================")
            self._sum_reward = 0


@RewardUnitFactory.register("active_heading_direction")
class RewardActiveHeadingDirection(RewardUnit):
    """
    Reward unit that calculates the reward based on the active heading direction of the robot.

    Args:
        reward_function (RewardFunction): The reward function to be used.
        r_angle (float, optional): Weight for difference between max deviation of heading direction and desired heading direction. Defaults to 0.6.
        theta_m (float, optional): Maximum allowable deviation of the heading direction. Defaults to np.pi/6.
        theta_min (int, optional): Minimum allowable deviation of the heading direction. Defaults to 1000.
        ped_min_dist (float, optional): Minimum distance to pedestrians. Defaults to 8.0.
        iters (int, optional): Number of iterations to find a reachable available theta. Defaults to 60.
        _on_safe_dist_violation (bool, optional): Flag indicating whether to penalize the reward on safe distance violation. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _r_angle (float): Desired heading direction in the robot's local frame.
        _theta_m (float): Maximum allowable deviation of the heading direction.
        _theta_min (int): Minimum allowable deviation of the heading direction.
        _ped_min_dist (float): Minimum application distance to pedestrians.
        _iters (int): Number of iterations to find a reachable available theta.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        r_angle: float = 0.6,
        theta_m: float = np.pi / 6,
        theta_min: int = 1000,
        ped_min_dist: float = 8.0,
        iters: int = 60,
        _on_safe_dist_violation: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._r_angle = r_angle
        self._theta_m = theta_m
        self._theta_min = theta_min
        self._ped_min_dist = ped_min_dist
        self._iters = iters

    def __call__(
        self,
        goal_in_robot_frame: np.ndarray,
        action: np.ndarray,
        relative_location: np.ndarray,
        relative_x_vel: np.ndarray,
        relative_y_vel: np.ndarray,
        *args,
        **kwargs,
    ) -> float:
        """
        Calculates the reward based on the active heading direction of the robot.

        Args:
            goal_in_robot_frame (np.ndarray): The goal position in the robot's frame of reference.
            action (np.ndarray): The last action taken by the robot.
            relative_location (np.ndarray): The relative location of the pedestrians.
            relative_x_vel (np.ndarray): The relative x-velocity of the pedestrians.
            relative_y_vel (np.ndarray): The relative y-velocity of the pedestrians.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: The calculated reward based on the active heading direction.
        """
        if (
            relative_location is None
            or relative_x_vel is None
            or relative_y_vel is None
        ):
            return 0.0

        # prefer goal theta:
        theta_pre = goal_in_robot_frame[1]
        d_theta = theta_pre

        v_x = action[0]

        # get the pedestrian's position:
        if len(relative_location) != 0:  # tracker results
            d_theta = np.pi / 2  # theta_pre
            theta_min = self._theta_min
            for _ in range(self._iters):
                theta = random.uniform(-np.pi, np.pi)
                free = True
                for ped_location, ped_x_vel, ped_y_vel in zip(
                    relative_location, relative_x_vel, relative_y_vel
                ):
                    p_x = ped_location[0]
                    p_y = ped_location[1]
                    p_vx = ped_x_vel
                    p_vy = ped_y_vel

                    ped_dis = np.linalg.norm([p_x, p_y])

                    if ped_dis <= self._ped_min_dist:
                        ped_theta = np.arctan2(p_y, p_x)

                        # 3*robot_radius:= estimation for sum of the pedestrian radius and the robot radius
                        vector = ped_dis**2 - (3 * self.robot_radius) ** 2
                        if vector < 0:
                            continue  # in this case the robot likely crashed into the pedestrian, disregard this pedestrian

                        vo_theta = np.arctan2(
                            3 * self.robot_radius,
                            np.sqrt(vector),
                        )
                        # Check if the robot's trajectory intersects with the pedestrian's VO cone
                        theta_rp = np.arctan2(
                            v_x * np.sin(theta) - p_vy, v_x * np.cos(theta) - p_vx
                        )
                        if theta_rp >= (ped_theta - vo_theta) and theta_rp <= (
                            ped_theta + vo_theta
                        ):
                            free = False
                            break

                # Find the reachable available theta that minimizes the difference from the goal theta
                if free:
                    theta_diff = (theta - theta_pre) ** 2
                    if theta_diff < theta_min:
                        theta_min = theta_diff
                        d_theta = theta

        else:  # no obstacles:
            d_theta = theta_pre

        return self._r_angle * (self._theta_m - abs(d_theta))


@RewardUnitFactory.register("ped_type_safety_distance")
class RewardPedTypeSafetyDistance(RewardUnit):
    """
    RewardPedTypeDistance is a reward unit that provides a reward based on the distance between the agent and a specific pedestrian type.

    Args:
        reward_function (RewardFunction): The reward function to which this reward unit belongs.
        ped_type (int, optional): The type of pedestrian to consider. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE.
        reward (float, optional): The reward value to be added if the distance to the pedestrian type is less than the safety distance. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD.
        safety_distance (float, optional): The safety distance threshold. If the distance to the pedestrian type is less than this value, the reward is added. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE.
        _on_safe_dist_violation (bool, optional): A flag indicating whether to trigger a violation event when the safety distance is violated. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _type (int): The type of pedestrian to consider.
        _reward (float): The reward value to be added if the distance to the pedestrian type is less than the safety distance.
        _safety_distance (float): The safety distance threshold.

    Methods:
        __call__(*args, **kwargs): Calculates the reward based on the distance to the pedestrian type.
        reset(): Resets the reward unit.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.REWARD,
        safety_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._reward = reward
        self._safety_distance = safety_distance

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        ped_type_min_distances = self.get_internal_state_info(
            "min_distances_per_ped_type"
        )

        if ped_type_min_distances is None:
            self.add_internal_state_info(
                key="min_distances_per_ped_type",
                value=get_ped_type_min_distances(**kwargs),
            )
            ped_type_min_distances = self.get_internal_state_info(
                "min_distances_per_ped_type"
            )

        if self._type not in ped_type_min_distances:
            rospy.logwarn(
                f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {self._type} not found."
            )
            return

        if ped_type_min_distances[self._type] < self._safety_distance:
            self.add_reward(self._reward)

    def reset(self):
        pass


@RewardUnitFactory.register("ped_type_collision")
class RewardPedTypeCollision(RewardUnit):
    """
    RewardPedTypeCollision is a reward unit that provides a reward when the robot collides with a specific pedestrian type.

    Args:
        reward_function (RewardFunction): The reward function to which this reward unit belongs.
        ped_type (int, optional): The specific pedestrian type to check for collision. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE.
        reward (float, optional): The reward value to be added when a collision with the specific pedestrian type occurs. Defaults to DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _type (int): The specific pedestrian type to check for collision.
        _reward (float): The reward value to be added when a collision with the specific pedestrian type occurs.
    """

    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.TYPE,
        reward: float = DEFAULTS.PED_TYPE_SPECIFIC_COLLISION.REWARD,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, True, *args, **kwargs)
        self._type = ped_type
        self._reward = reward

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Checks if the robot has collided with the specific pedestrian type and adds the reward if a collision occurs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ped_type_min_distances = self.get_internal_state_info(
            "min_distances_per_ped_type"
        )

        if ped_type_min_distances is None:
            self.add_internal_state_info(
                key="min_distances_per_ped_type",
                value=get_ped_type_min_distances(**kwargs),
            )
            ped_type_min_distances = self.get_internal_state_info(
                "min_distances_per_ped_type"
            )

        if self._type not in ped_type_min_distances:
            rospy.logwarn(
                f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {self._type} not found."
            )
            return

        if ped_type_min_distances[self._type] <= self.robot_radius:
            self.add_reward(self._reward)

    def reset(self):
        pass


@RewardUnitFactory.register("ped_type_vel_constraint")
class RewardPedTypeVelocityConstraint(RewardUnit):

    def __init__(
        self,
        reward_function: RewardFunction,
        ped_type: int = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.TYPE,
        penalty_factor: float = 0.05,
        active_distance: float = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE.DISTANCE,
        _on_safe_dist_violation: bool = DEFAULTS.PED_TYPE_SPECIFIC_SAFETY_DISTANCE._ON_SAFE_DIST_VIOLATION,
        *args,
        **kwargs,
    ):
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._type = ped_type
        self._penalty_factor = penalty_factor
        self._active_distance = active_distance

    def __call__(self, action: np.ndarray, *args: Any, **kwargs: Any) -> None:
        ped_type_min_distances = self.get_internal_state_info(
            "min_distances_per_ped_type"
        )

        if ped_type_min_distances is None:
            self.add_internal_state_info(
                key="min_distances_per_ped_type",
                value=get_ped_type_min_distances(**kwargs),
            )
            ped_type_min_distances = self.get_internal_state_info(
                "min_distances_per_ped_type"
            )

        if self._type not in ped_type_min_distances:
            rospy.logwarn(
                f"[{rospy.get_name()}, {self.__class__.__name__}] Pedestrian type {self._type} not found."
            )
            return

        if ped_type_min_distances[self._type] < self._active_distance:
            self.add_reward(-self._penalty_factor * action[0])

    def reset(self):
        pass

from scipy.spatial import cKDTree
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
@RewardUnitFactory.register("follow_tebplan")
class RewardFollowTebplan(RewardUnit):
    @check_params
    def __init__(
        self,
        reward_function: "RewardFunction",
        reward_factor: float = DEFAULTS.FOLLOW_GLOBALPLAN.REWARD_FACTOR,
        is_integral_distance: bool = True,
        _on_safe_dist_violation: bool = DEFAULTS.FOLLOW_GLOBALPLAN._ON_SAFE_DIST_VIOLATION,
        *args, **kwargs,
    ) -> None:
        super().__init__(reward_function, _on_safe_dist_violation, *args, **kwargs)
        self._reward_factor = reward_factor
        self.is_integral_distance = is_integral_distance
        self.is_normalize_points = rospy.get_param_cached("is_normalize_points", False)
        self.action_points_num = rospy.get_param_cached("action_points_num", 0)
        self.marker_pub = rospy.Publisher('Tebplan_visualization_marker', Marker, queue_size=1)
        # 根据self.action_points_num计算时间间隔的索引
        self.time_intervals = np.linspace(1, 10, self.action_points_num, dtype=int)
        self._step_size = rospy.get_param_cached("/step_size", 0.2)
        self.empty_count = 0

    def check_parameters(self, *args, **kwargs):
        """
        Checks if the reward value is positive and issues a warning if it is.
        """
        if not self.is_normalize_points:
            rospy.logerr("RewardActionPointsChange only use at Action points mode")
            raise ValueError("RewardActionPointsChange only use at Action points mode")
        
        if self.is_normalize_points and self.action_points_num == 0:
            rospy.logerr("Action points num is 0")
            raise ValueError("Action points num is 0")

    def __call__(
        self,
        action_points_map: np.ndarray,
        teb_plan: np.ndarray,
        robot_pose,
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        if teb_plan.size == 0 or action_points_map.size == 0:
            self.empty_count += 1
            if self.empty_count > 500:
                rospy.logerr("Empty teb_plan or action_points_map for 500 times")
            return
        self.empty_count = 0
        
        # check action_points_map and teb_plan shape
        if action_points_map.shape[1] != 2 or teb_plan.shape[1] != 2:
            rospy.logerr("Invalid shape of action_points_map or teb_plan")
            raise ValueError("Invalid shape of action_points_map or teb_plan")

        # self._kdtree = cKDTree(teb_plan)
        # _, nearest_idx = self._kdtree.query([robot_pose.x, robot_pose.y], k=1)
        # teb_plan = teb_plan[nearest_idx:]  # use only the part of the plan from nearest point to end
        try:
            # 反转TEB轨迹
            reversed_teb_plan = teb_plan[::-1]
            # 创建KD树用于查询
            kd_tree = cKDTree(reversed_teb_plan)
            # 查询机器人当前位置最近的点
            _, nearest_idx = kd_tree.query([robot_pose.x, robot_pose.y], k=1)
            # 计算原始轨迹中的索引
            original_idx = len(teb_plan) - 1 - nearest_idx
            # 裁剪原始轨迹从最近点到末尾
            teb_plan = teb_plan[original_idx:]

            # Assuming robot_pose is an object with attributes x and y
            robot_start = np.array([robot_pose.x, robot_pose.y])
            if self.is_integral_distance:
                # Compute distances efficiently
                distances = np.linalg.norm(np.diff(action_points_map, axis=0, prepend=[robot_start]), axis=1)
                # np.cumsum 函数计算给定数组的累积和,每个元素 accumulated_distances[i] 存储的是从机器人初始位置到动作点 action_points_map[i] 的总距离
                accumulated_distances = np.cumsum(distances)
            else:
                # 根据时间间隔和固定速度计算累积距离
                speed = 0.55  # m/s
                accumulated_distances = self.time_intervals * self._step_size * speed

            # Efficiently find corresponding points in the trimmed TEB plan
            teb_distances = np.linalg.norm(np.diff(teb_plan, axis=0, prepend=[robot_start]), axis=1)
            teb_accumulated = np.cumsum(teb_distances)

            # Efficiently map action points to closest TEB points
            idx = np.searchsorted(teb_accumulated, accumulated_distances)
            idx = np.clip(idx, 0, len(teb_plan) - 1)  # Ensure indices are within bounds

            # Compute the distance-based penalty as reward using vectorized operations
            distances_to_teb = np.linalg.norm(action_points_map - teb_plan[idx], axis=1)
            reward = -np.sum(distances_to_teb) * self._reward_factor

            self.add_reward(reward)
            if if_show_reward:
                self._sum_reward += reward

            # corresponding_teb_points = teb_plan[idx]
            # self.publish_markers(corresponding_teb_points)
        except Exception as e:
            rospy.logerr(f"Error in RewardFollowTebplan: {e}")
    
    def reset(self):
        if if_show_reward:
            print("FollowTebplan reward:", self._sum_reward)
            print("==================================================")
            self._sum_reward = 0

    def publish_markers(self, points):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "teb_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1
        marker.scale.x = 0.2  # 点的大小
        marker.scale.y = 0.2
        marker.color.a = 1.0  # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 0.5
        for p in points:
            pt = Point()
            pt.x = p[0]
            pt.y = p[1]
            pt.z = 0
            marker.points.append(pt)
        self.marker_pub.publish(marker)