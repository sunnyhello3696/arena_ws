import numpy as np
from gymnasium import spaces
from numpy import ndarray
from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS

from ...observation_space_factory import SpaceFactory
from ..base_observation_space import BaseObservationSpace

import math
from collections import deque

import numpy as np
import rospy  # Assuming use of ROS for rviz
from sensor_msgs.msg import Image  # Assuming conversion to Image message for rviz
import cv2  # 导入OpenCV库
from flatland_msgs.msg import Galaxy2D
from typing import Any, Dict

# import ros_numpy


@SpaceFactory.register("convexqueue")
class ConvexQueueSpace(BaseObservationSpace):
    """
    Represents the observation space for laser scan data.

    Args:
        laser_num_beams (int): The number of laser beams.
        laser_max_range (float): The maximum range of the laser.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _num_beams (int): The number of laser beams.
        _max_range (float): The maximum range of the laser.
    """

    def __init__(
        self, 
        laser_num_beams: int,
        laser_max_range: float,
        *args, 
        **kwargs,
    ) -> None:
        self._laser_num_beams = laser_num_beams
        self._laser_max_range = laser_max_range + 0.25
        self._max_vertex_num = rospy.get_param_cached("/max_vertex_num", 120)

        super().__init__(*args, **kwargs)

        print("ConvexQueueSpace init")
        rospy.loginfo("ConvexQueueSpace init")

    @property
    def space(self) -> spaces.Box:
        """
        Get the gym.Space object representing the observation space.
        """
        return self._space

    def get_gym_space(self) -> spaces.Box:
        """
        Returns the Gym observation space for laser scan data.

        Returns:
            spaces.Space: The Gym observation space.
        """
        return spaces.Box(
            low=0,
            high=self._laser_max_range,
            shape=(self._max_vertex_num,),
            dtype=np.float32,
        )

    @BaseObservationSpace.apply_normalization
    def encode_observation(self, observation: Dict[str, Any], *args, **kwargs) -> ndarray:
        """
        Encodes the laser scan observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded laser scan observation.
        """
        # observation[OBS_DICT_KEYS.LASER_CONVEX]: Galaxy2D
        laser_convex: Galaxy2D = observation[OBS_DICT_KEYS.LASER_CONVEX]

        # observation[OBS_DICT_KEYS.LASER]: np.array(sensor_msgs.msg.LaserScan.ranges, np.float32)
        laser_scan = observation[OBS_DICT_KEYS.LASER]
        
        g2d_cal_success = laser_convex.success.data
        g2d_polar_convex = laser_convex.polar_convex
        g2d_polar_convex_theta = laser_convex.polar_convex_theta

        if g2d_cal_success and len(g2d_polar_convex) == self._max_vertex_num:
            process_scan = g2d_polar_convex
        else:
            rospy.logwarn("ConvexQueueSpace: Galaxy2D convex failed, using laser scan")
            g2d_cal_success = False
            process_scan = []
            # g2d_convex_vertex = []
            g2d_polar_convex = []
            # g2d_polar_convex_theta = [i*2*np.pi/self._max_vertex_num for i in range(self._max_vertex_num)]
            # 0-359
            di = int(self._laser_num_beams/self._max_vertex_num)
            for i in range(self._max_vertex_num):
                # g2d_convex_vertex.append((laser_scan[di*i]*np.cos(g2d_polar_convex_theta[i]),laser_scan[di*i]*np.sin(g2d_polar_convex_theta[i])))
                g2d_polar_convex.append(laser_scan[di*i])
                process_scan.append(laser_scan[di*i])
                # process_scan.append(g2d_polar_convex_theta[i]/2./np.pi)
        
        # Convert list to NumPy array
        process_scan_array = np.array(process_scan)

        # Find indices where values are NaN or Inf
        nan_indices = np.isnan(process_scan_array)
        inf_indices = np.isinf(process_scan_array)

        if np.any(nan_indices) or np.any(inf_indices):
            rospy.logwarn("ConvexQueueSpace: NaN or Inf values detected in laser scan")

            # Replace NaN and Inf values with laser_max_range
            process_scan_array[nan_indices] = self._laser_max_range
            process_scan_array[inf_indices] = self._laser_max_range

        return process_scan_array
