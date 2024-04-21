from collections import deque

import numpy as np
import numpy.matlib
import rospy
from gymnasium import spaces
from numpy import ndarray

from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from .base_feature_map_space import BaseFeatureMapSpace


@SpaceFactory.register("stacked_laser_map")
class StackedLaserMapSpace(BaseFeatureMapSpace):
    """
    Represents a feature map space for stacked laser maps.

    Args:
        laser_stack_size (int): The size of the laser stack.
        feature_map_size (int): The size of the feature map.
        roi_in_m (float): The region of interest in meters.
        flatten (bool, optional): Whether to flatten the feature map. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _laser_queue (deque): A queue to store the laser scans.
        _laser_stack_size (int): The size of the laser stack.

    Methods:
        _reset_laser_stack(laser_scan: np.ndarray): Resets the laser stack with zeros.
        _process_laser_scan(laser_scan: np.ndarray, done: bool) -> np.ndarray: Processes the laser scan and returns the feature map.
        _build_laser_map(laser_queue: deque) -> np.ndarray: Builds the laser map from the laser queue.
        get_gym_space() -> spaces.Space: Returns the gym space for the feature map.
        encode_observation(observation: dict, *args, **kwargs) -> ndarray: Encodes the observation into a feature map.

    """

    def __init__(
        self,
        laser_stack_size: int,
        feature_map_size: int,
        roi_in_m: float,
        flatten: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self._laser_queue = deque()
        self._laser_stack_size = laser_stack_size
        self._default_reward_info = {}
        super().__init__(
            feature_map_size=feature_map_size,
            roi_in_m=roi_in_m,
            flatten=flatten,
            *args,
            **kwargs,
        )

    def _reset_laser_stack(self, laser_scan: np.ndarray):
        """
        Resets the laser stack with zeros.

        Args:
            laser_scan (np.ndarray): The laser scan.

        """
        self._laser_queue = deque([np.zeros_like(laser_scan)] * self._laser_stack_size)

    def _process_laser_scan(self, laser_scan: np.ndarray, done: bool) -> np.ndarray:
        """
        Processes the laser scan and returns the feature map.

        Args:
            laser_scan (np.ndarray): The laser scan.
            done (bool): Whether the episode is done.

        Returns:
            np.ndarray: The feature map.

        """
        if type(laser_scan) is not np.ndarray:
            print("laser_scan is not np.ndarray")
            return np.zeros((self._feature_map_size * self._feature_map_size,))

        if len(self._laser_queue) == 0:
            self._reset_laser_stack(laser_scan)

        # 将当前的激光扫描数据添加到队列的前端，同时移除队列末尾的数据，以维持队列（激光堆栈）的大小不变。
        # 这样，队列始终保持了最新的激光扫描数据，而最旧的数据被逐渐淘汰。
        self._laser_queue.pop()
        self._laser_queue.appendleft(laser_scan)

        laser_map = self._build_laser_map(self._laser_queue)

        if done:
            self._reset_laser_stack(laser_scan)

        return laser_map

    def _build_laser_map(self, laser_queue: deque) -> np.ndarray:
        """
        Builds the laser map from the laser queue.

        Args:
            laser_queue (deque): The laser queue.

        Returns:
            np.ndarray: The laser map.

        """
        try:
            # laser_array = np.array(laser_queue)
            # # laserstack list of 10 np.arrays of shape (720,)
            # scan_avg = np.zeros((20, self._feature_map_size))
            # # horizontal stacking of the pooling operations
            # # min pooling over every 9th entry
            # scan_avg[::2, :] = np.min(
            #     laser_array.reshape(10, self._feature_map_size, 9), axis=2
            # )
            # # avg pooling over every 9th entry
            # scan_avg[1::2, :] = np.mean(
            #     laser_array.reshape(10, self._feature_map_size, 9), axis=2
            # )

            # scan_avg_map = np.tile(scan_avg.ravel(), 4).reshape(
            #     (self._feature_map_size, self._feature_map_size)
            # )
            
            # 将所有的激光扫描数据转换成一个一维数组。
            temp = np.array(laser_queue, dtype=np.float32).flatten()

            # 初始化一个空的特征映射矩阵 scan_avg，其大小为 (20, 80)。这个矩阵用于存放处理后的激光数据，即经过最小化和平均化处理的数据。
            scan_avg = np.zeros((10, 80))


            for n in range(5):
                scan_tmp = temp[n * 720 : (n + 1) * 720]
                for i in range(80):
                    scan_avg[2 * n, i] = np.min(scan_tmp[i * 9 : (i + 1) * 9])
                    scan_avg[2 * n + 1, i] = np.mean(scan_tmp[i * 9 : (i + 1) * 9])

            scan_avg = scan_avg.reshape(800)
            # 将一个由激光扫描数据处理得到的一维数组 scan_avg 重复扩展，并重新整形为一个 (80, 80) 形状的二维数组
            scan_avg_map = np.matlib.repmat(scan_avg, 2, 4).reshape((80, 80))
        except Exception as e:
            rospy.logwarn(
                f"[{rospy.get_name()}, {StackedLaserMapSpace.__name__}]: {e} \n Cannot build laser map. Instead return empty map."
            )
            return np.zeros(self.get_gym_space().shape)

        return scan_avg_map

    def get_gym_space(self) -> spaces.Space:
        """
        Returns the gym space for the feature map.

        Returns:
            spaces.Space: The gym space.

        """
        return spaces.Box(
            low=0,
            high=self._roi_in_m,
            shape=(self._feature_map_size * self._feature_map_size,),
            dtype=np.float32,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        """
        Encodes the observation into a feature map.

        Args:
            observation (dict): The observation dictionary.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ndarray: The encoded feature map.

        """
        return self._process_laser_scan(
            observation[OBS_DICT_KEYS.LASER],
            kwargs.get(OBS_DICT_KEYS.DONE, False),
        ).flatten()
