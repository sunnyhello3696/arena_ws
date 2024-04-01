from typing import List
from typing import Any, Dict

import numpy as np
import rospy
from gymnasium import spaces

from ..utils.action_space.action_space_manager import ActionSpaceManager
from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import SPACE_INDEX
from .base_space_encoder import BaseSpaceEncoder
from .encoder_factory import BaseSpaceEncoderFactory
# from rl_utils.utils.observation_collector.constants import OBS_DICT_KEYS
# from rl_utils.utils.observation_collector.utils import pose3d_to_pose2d
from flatland_msgs.msg import Galaxy2D
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, PoseStamped, Pose
from geometry_msgs.msg import Twist
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.interpolate import splprep, splev

"""
    This encoder offers a robot specific observation and action space
    Different actions spaces for holonomic and non holonomic robots

    Observation space:   Laser Scan, Goal, Current Vel 
    Action space: X Vel, (Y Vel), Angular Vel

"""


@BaseSpaceEncoderFactory.register("ConvexMPCEncoder")
class ConvexMPCEncoder(BaseSpaceEncoder):
    """
    DefaultEncoder class is responsible for encoding and decoding actions and observations
    using the default action and observation space managers.
    """

    DEFAULT_OBS_LIST = [
        SPACE_INDEX.CONVEX,
        SPACE_INDEX.GOAL,
        SPACE_INDEX.LAST_ACTION,
    ]

    def __init__(
        self,
        action_space_kwargs: dict,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
        *args,
        **kwargs
    ):
        """
        Initializes a new instance of the DefaultEncoder class.

        Args:
            action_space_kwargs (dict): Keyword arguments for configuring the action space manager.
            observation_list (List[SPACE_INDEX], optional): List of observation spaces to include. Defaults to None.
            observation_kwargs (dict, optional): Keyword arguments for configuring the observation space manager. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        merged_kwargs = {}
        merged_kwargs.update(action_space_kwargs)
        merged_kwargs.update(observation_kwargs)
        super().__init__(**merged_kwargs, **kwargs)
        self._observation_list = observation_list
        self._observation_kwargs = observation_kwargs
        self.setup_action_space(action_space_kwargs)
        self.setup_observation_space(observation_list, observation_kwargs)
        self.spline_pub = rospy.Publisher('action_points_spline', Marker, queue_size=1)
        print("ConvexMPCEncoder init")
        rospy.loginfo("ConvexMPCEncoder init")

    @property
    def observation_space(self) -> spaces.Space:
        """
        Gets the observation space.

        Returns:
            spaces.Space: The observation space.
        """
        return self._observation_space_manager.observation_space

    @property
    def action_space(self) -> spaces.Space:
        """
        Gets the action space.

        Returns:
            spaces.Space: The action space.
        """
        return self._action_space_manager.action_space

    @property
    def action_space_manager(self):
        """
        Gets the action space manager.

        Returns:
            ActionSpaceManager: The action space manager.
        """
        return self._action_space_manager

    @property
    def observation_space_manager(self):
        """
        Gets the observation space manager.

        Returns:
            ObservationSpaceManager: The observation space manager.
        """
        return self._observation_space_manager

    @property
    def observation_list(self):
        """
        Gets the list of observation spaces.

        Returns:
            List[SPACE_INDEX]: The list of observation spaces.
        """
        return self._observation_list

    @property
    def observation_kwargs(self):
        """
        Gets the keyword arguments for configuring the observation space manager.

        Returns:
            dict: The keyword arguments for configuring the observation space manager.
        """
        return self._observation_kwargs

    def setup_action_space(self, action_space_kwargs: dict):
        """
        Sets up the action space manager.

        Args:
            action_space_kwargs (dict): Keyword arguments for configuring the action space manager.
        """
        self._action_space_manager = ActionSpaceManager(**action_space_kwargs)

    def setup_observation_space(
        self,
        observation_list: List[SPACE_INDEX] = None,
        observation_kwargs: dict = None,
    ):
        """
        Sets up the observation space manager.

        Args:
            observation_list (List[SPACE_INDEX], optional): List of observation spaces to include. Defaults to None.
            observation_kwargs (dict, optional): Keyword arguments for configuring the observation space manager. Defaults to None.
        """
        if not observation_list:
            observation_list = self.DEFAULT_OBS_LIST

        self._observation_space_manager = ObservationSpaceManager(
            observation_list,
            space_kwargs=observation_kwargs,
            frame_stacking=self._stacked,
        )

    def decode_action(self, action, action_obs_dict: Dict[str, Any]) -> np.ndarray:
        """
        Decodes the action.

        Args:
            action: The action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        action_points = [] # base下
        polar_action = []
        netout_scale_factors = self._action_space_manager.decode_action(action)
        if action_obs_dict is not None:
            self.last_obs_dict = self.get_observations_d86(action_obs_dict)
            if self.last_obs_dict["laser_convex"][0] is not None:
                rvx,rvy = self.worldva2robotva(self.last_obs_dict["robot_world_vel"].x,self.last_obs_dict["robot_world_vel"].y,self.last_obs_dict["robot_world_pose"].theta)
                
                action_points.append((self._robot_pose.x,self._robot_pose.y))
                # 输入的速度方向为 激光雷达0度坐标系下的方向
                # 网络输出的点：xy坐标点action_points、极坐标点polar_action
                one_action_point1,one_polar_action1 = self.calc_polar_action_points((0.,0.,np.arctan2(rvy,rvx)),(netout_scale_factors[0],netout_scale_factors[1]),0)
                polar_action.append(one_polar_action1)
                action_points.append(one_action_point1)

                one_action_point2,one_polar_action2 = self.calc_polar_action_points((action_points[1][0],action_points[1][1],np.arctan2(action_points[1][1],action_points[1][0])),(netout_scale_factors[2],netout_scale_factors[3]),1)
                polar_action.append(one_polar_action2)
                action_points.append(one_action_point2)


                # 假设 action_points 包含您从上面代码片段中计算出的点
                action_points = np.array(action_points, dtype=np.float32)

                # 打印action_points的形状和内容来进行检查
                print("action_points shape:", action_points.shape)
                print("action_points content:", action_points)

                # 确保点的数量足够
                if action_points.shape[0] < 3:
                    raise ValueError("至少需要3个点来进行2次B-Spline拟合")

                # 确保点的维度正确（这里我们假设是二维点，即n=2）
                if action_points.ndim != 2 or action_points.shape[1] != 2:
                    raise ValueError("点的格式应为(m, 2)，其中m是点的数量")
                
                # 检查数据点是否包含NaN或无穷值
                if np.isnan(action_points).any() or np.isinf(action_points).any():
                    raise ValueError("数据点包含NaN或无穷值")

                # 尝试使用不同的平滑参数
                s = 2  # 您可以根据需要调整这个参数

                try:
                    tck, u = splprep(action_points.T, s=s, per=False, k=2)
                    u_new = np.linspace(u.min(), u.max(), 100)
                    x_new, y_new = splev(u_new, tck, der=0)
                    self.publish_spline(x_new, y_new)
                except Exception as e:
                    print("拟合过程中发生错误:", e)
                    # 根据错误类型进行适当的处理
                    return None
                     
        return self._action_space_manager.decode_action(action)

    def encode_observation(self, observation, *args, **kwargs) -> np.ndarray:
        """
        Encodes the observation.

        Args:
            observation: The observation to encode.

        Returns:
            np.ndarray: The encoded observation.
        """
        return self._observation_space_manager.encode_observation(observation, **kwargs)

    def get_observations_d86(self, action_obs_dict: Dict[str, Any]):
        msg_galaxy2d: Galaxy2D= action_obs_dict["laser_convex"]
        state: Odometry = action_obs_dict["robot_state"]

        self.laser_num_beams = 360
        self.max_vertex_num = 360

        robot_state= self.process_robot_state_msg(state)

        self._robot_pose = robot_state[0]
        self._robot_vel = robot_state[1]

        g2d_cal_success = msg_galaxy2d.success.data

        if msg_galaxy2d.scans is None or len(msg_galaxy2d.scans) != self.laser_num_beams:
            print(f"Galaxy2D scan is None or not complete {self.laser_num_beams} degrees.")
            scans = [0. for i in range(self.laser_num_beams)]
        else:
            scans = msg_galaxy2d.scans

        g2d_convex_vertex = []
        for p in msg_galaxy2d.convex_vertex.points:
            g2d_convex_vertex.append((p.x,p.y))

        g2d_polar_convex = msg_galaxy2d.polar_convex
        g2d_polar_convex_theta = msg_galaxy2d.polar_convex_theta

        local_feasible_space,global_feasible_space = \
        self.feasible_position(scans,g2d_cal_success,g2d_polar_convex)

        if g2d_cal_success and len(g2d_polar_convex) == self.max_vertex_num:
            process_scan = []
        else:
            g2d_cal_success = False
            g2d_convex_vertex = []
            g2d_polar_convex = []
            g2d_polar_convex_theta = [i*2*np.pi/self.max_vertex_num for i in range(self.max_vertex_num)]
            # 0-359
            di = int(self.laser_num_beams/self.max_vertex_num)
            for i in range(self.max_vertex_num):
                g2d_convex_vertex.append((scans[di*i]*np.cos(g2d_polar_convex_theta[i]),scans[di*i]*np.sin(g2d_polar_convex_theta[i])))
                g2d_polar_convex.append(scans[di*i])

        obs_dict_d86 = {
            "laser_scan": np.array(scans),
            "robot_world_pose": self._robot_pose,
            "robot_world_vel": self._robot_vel.linear,
            "feasible_space":(local_feasible_space,global_feasible_space),
            "laser_convex":[g2d_convex_vertex,g2d_polar_convex,g2d_polar_convex_theta,g2d_cal_success],
        }

        return obs_dict_d86


    def feasible_position(self,scans,g2d_cal_success,g2d_polar_convex):
        local_feasible_dist = -10.
        # global_feasible_dist = -10.
        # local_feasible_dist = 10.
        plan_dynamic_limit:tuple=(2.5,2.,4.)

        global_feasible_dist = 10.
        vthe = np.arctan2(self._robot_vel.linear.y,self._robot_vel.linear.x)
        athe = np.arctan2(self._robot_vel.angular.y,self._robot_vel.angular.x)
        jerk = []
        jerk_max = plan_dynamic_limit[2]*1.41421356
        jerk.append((jerk_max*np.cos(vthe),jerk_max*np.sin(vthe)))
        jerk.append((jerk_max*np.cos(athe),jerk_max*np.sin(athe)))
        jerk.append((jerk_max*np.cos((vthe+athe)/2.),jerk_max*np.sin((vthe+athe)/2.)))
        jerk.append((-jerk_max*np.cos((vthe+athe)/2.),-jerk_max*np.sin((vthe+athe)/2.)))
        
        for j in jerk:
            x = self._robot_pose.x + self._robot_vel.linear.x*0.1 + self._robot_vel.angular.x*0.1**2/2. + j[0]*0.1**3/6.
            y = self._robot_pose.y + self._robot_vel.linear.y*0.1 + self._robot_vel.angular.y*0.1**2/2. + j[1]*0.1**3/6.
            d = np.hypot(x-self._robot_pose.x,y-self._robot_pose.y)
            if d > local_feasible_dist:
                local_feasible_dist = d
        local_feasible_dist = 3*local_feasible_dist

        for j in jerk:
            x = self._robot_pose.x + self._robot_vel.linear.x + self._robot_vel.angular.x/2. + j[0]/6.
            y = self._robot_pose.y + self._robot_vel.linear.y + self._robot_vel.angular.y/2. + j[1]/6.
            d = np.hypot(x-self._robot_pose.x,y-self._robot_pose.y)
            if d > global_feasible_dist:
                global_feasible_dist = d

        local_feasible_space = []
        global_feasible_space = []
        if g2d_cal_success:
            for i in range(len(g2d_polar_convex)):
                local_feasible_space.append(min(g2d_polar_convex[i],local_feasible_dist))
                global_feasible_space.append(min(g2d_polar_convex[i],global_feasible_dist))
        else:
            for i in range(len(scans)):
                local_feasible_space.append(min(scans[i],local_feasible_dist))
                global_feasible_space.append(min(scans[i],global_feasible_dist))
        return local_feasible_space,global_feasible_space


    def process_robot_state_msg(self, msg_Odometry: Odometry): 
        pose3d = msg_Odometry.pose.pose
        twist = msg_Odometry.twist.twist
        return self.pose3D_to_pose2D(pose3d), twist
    
    def pose3D_to_pose2D(self, pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w,
        )
        euler = self.euler_from_quaternion(*quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d
    

    def euler_from_quaternion(self,x, y, z, w):
        euler = [0, 0, 0]
        Epsilon = 0.0009765625
        Threshold = 0.5 - Epsilon
        TEST = w * y - x * z
        if TEST < -Threshold or TEST > Threshold:
            if TEST > 0:
                sign = 1
            elif TEST < 0:
                sign = -1
            euler[2] = -2 * sign * math.atan2(x, w)
            euler[1] = sign * (math.pi / 2.0)
            euler[0] = 0
        else:
            euler[0] = math.atan2(2 * (y * z + w * x),
                                    w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z),
                                    w * w + x * x - y * y - z * z)

        return euler
    
    def worldva2robotva(self, wvax,wvay,now_wyaw):
        va =np.hypot(wvax,wvay)
        vayaw = np.arctan2(wvay,wvax)
        vax = np.cos(vayaw - now_wyaw)*va
        vay = np.sin(vayaw - now_wyaw)*va
        return vax,vay

    def calc_polar_action_points(self,start,action,whcich):
        """

        """
        theta = self.NormalizeAngleTo2Pi(2*np.pi*action[0] + start[2])
        min_index = 0
        min_theta = 3*np.pi
        thetas = []
        convex_scans_limit_range = self.last_obs_dict["feasible_space"][whcich]
        for i in range(self.max_vertex_num):
            p = (convex_scans_limit_range[i] * np.cos(self.last_obs_dict["laser_convex"][2][i]), \
                convex_scans_limit_range[i] * np.sin(self.last_obs_dict["laser_convex"][2][i]))
            thetas.append(self.NormalizeAngleTo2Pi(np.arctan2(p[1] - start[1], p[0] - start[0])))
            if thetas[-1] < min_theta:
                min_theta = thetas[-1]
                min_index = i

        _find = False
        next = min_index
        prev = min_index-1
        for j in range(min_index+1):
            if theta < thetas[j]:
                next = j
                prev = next - 1
                _find = True
                break
        if not _find:
            for j in range(min_index,self.max_vertex_num):
                if theta < thetas[j]:
                    next = j
                    prev = next - 1
                    break

        crossp = self.calc_cross_point(start, \
                                (start[0] + 2 * self._laser_max_range * np.cos(theta), \
                                start[1] + 2 * self._laser_max_range * np.sin(theta)), \
                                (convex_scans_limit_range[prev] * np.cos(
                                    self.last_obs_dict["laser_convex"][2][prev]), \
                                convex_scans_limit_range[prev] * np.sin(
                                    self.last_obs_dict["laser_convex"][2][prev])), \
                                (convex_scans_limit_range[next] * np.cos(
                                    self.last_obs_dict["laser_convex"][2][next]), \
                                convex_scans_limit_range[next] * np.sin(
                                    self.last_obs_dict["laser_convex"][2][next])))
        dist = np.hypot(crossp[0] - start[0], crossp[1] - start[1]) * action[1]
        action_point = (start[0] + dist * np.cos(theta), start[1] + dist * np.sin(theta))
        polar_point = (np.arctan2(action_point[1],action_point[0]),np.hypot(action_point[1],action_point[0]))
        
        
        # print("action")
        # print(action)
        # print("theta")
        # print(self.last_obs_dict["laser_convex"][2])
        # print(theta)
        # print("dist and  crossp")
        # print(self.last_obs_dict["laser_convex"][1])
        # print(crossp)
        # print("polar res    ")
        # print((theta,dist))
        return  action_point,polar_point
    
    def NormalizeAngleTo2Pi(self, d_theta):# 0-2pi
        d_theta_normalize = d_theta
        while d_theta_normalize > 2 * math.pi:
            d_theta_normalize = d_theta_normalize - 2 * math.pi
        while d_theta_normalize < 0:
            d_theta_normalize = d_theta_normalize + 2 * math.pi
        return d_theta_normalize
    
    @staticmethod
    def calc_cross_point(a1,a2,b1,b2):
        def cross(a,b):
            return a[0]*b[1]-a[1]*b[0] 
        a = (a2[0]-a1[0],a2[1]-a1[1])
        b = (b2[0]-b1[0],b2[1]-b1[1])
        frac = cross((b1[0]-a1[0],b1[1]-a1[1]),b)
        num = cross(a,b)
        t = frac/num
        return (a1[0]+t*a[0],a1[1]+t*a[1])
    
    def publish_spline(self,x_new, y_new):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.02  # 线宽
        marker.color.a = 1.0  # 不透明度
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        # 假设x_new和y_new是通过SciPy得到的曲线拟合点
        for x, y in zip(x_new, y_new):
            p = Point()
            p.x = x
            p.y = y
            p.z = 0
            marker.points.append(p)
        self.spline_pub.publish(marker)
        return