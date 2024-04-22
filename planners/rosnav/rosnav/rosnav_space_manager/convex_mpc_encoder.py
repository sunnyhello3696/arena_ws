from typing import List
from typing import Any, Dict

import numpy as np
import rospy
from gymnasium import spaces

from ..utils.action_space.action_space_manager import ActionSpaceManager
from ..utils.observation_space.observation_space_manager import ObservationSpaceManager
from ..utils.observation_space.space_index import SPACE_INDEX
from ..utils.unicycle_mpc_x2_u0 import ConvexUnicycleMPC
from ..utils.unicycle_mpc_convex_qpsolvers import ConvexUnicycleMPC_qpsolver
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
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
import yaml
from numpy import genfromtxt

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
        super().__init__(action_space_kwargs,observation_list,observation_kwargs, **kwargs)
        merged_kwargs = {}
        merged_kwargs.update(action_space_kwargs)
        merged_kwargs.update(observation_kwargs)
        self.is_normalize_points = merged_kwargs.get("normalize_points", False)
        self.action_points_num = merged_kwargs.get("action_points_num", 0)
        if self.is_normalize_points and self.action_points_num > 0:
            self.setup_mpc_cotroller()
        self._step_size = rospy.get_param_cached("/step_size", 0.2)
        print("step_size:",self._step_size)
        self._max_vertex_num = rospy.get_param_cached("/max_vertex_num", 120)
        self.obs_dict_d86 = None
        self.laser_num_beams = 360
        self._laser_max_range = merged_kwargs.get("laser_max_range", 8.0)
        self._robot_pose = Pose2D()
        self._robot_vel = None
        self._last_action_points = None

        self.debug_vis = True
        self.mpc_mapframe_test_traj = True
        # if self.mpc_mapframe_test_traj:
            # self.mpc_xref_traj = genfromtxt("/home/dmz/Documents/mpc_test_traj/ref_states_2.csv", delimiter=',')
            # self.mpc_xref_traj = self.mpc_xref_traj[:,:2] # x,y
            # initial_pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)

        self.enable_rviz = False
        if rospy.get_param("/debug_mode", False):
            self.spline_pub = rospy.Publisher('action_points_spline', Marker, queue_size=1)
            self.action_points_pub = rospy.Publisher('action_points', Marker, queue_size=1)
            self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
            self.enable_rviz = rospy.get_param("/if_viz", False)
        print("ConvexMPCEncoder init")
        rospy.loginfo("ConvexMPCEncoder init")

        # time.sleep(10)

    def setup_mpc_cotroller(self):
        cfg_file_path = rospy.get_param("/mpc_config_path")
        cfg = {}
        with open(cfg_file_path, 'r') as stream:
            cfg = yaml.safe_load(stream)

        # print("cfg:",cfg)
        T = cfg['mpc']['T']
        N = int(cfg['mpc']['N'])
        self.k = int(cfg['mpc']['k'])
        self.if_add_robot_pt = bool(cfg['mpc']['if_add_robot_pt'])
        self._feasible_position_speed_factor = float(cfg['mpc']['feasible_position_speed_factor'])
        self.goal_in_convex_if_angle = bool(cfg['mpc']['goal_in_convex_if_angle'])
        xmin = np.array(cfg['mpc']['xmin1']).astype(np.float32)
        xmax = np.array(cfg['mpc']['xmax1']).astype(np.float32)
        umin = np.array(cfg['mpc']['umin1']).astype(np.float32)
        umax = np.array(cfg['mpc']['umax1']).astype(np.float32)

        Q = np.array(cfg['mpc']['Q1']).astype(np.float32)
        QN = np.array(cfg['mpc']['QN1']).astype(np.float32)
        R = np.array(cfg['mpc']['R1']).astype(np.float32)
        # self.mpc = ConvexUnicycleMPC(T, N, xmin, xmax, umin, umax, Q, QN, R)
        self.mpc = ConvexUnicycleMPC_qpsolver(T, N, xmin, xmax, umin, umax, Q, QN, R)

    def process_action(self, action, action_obs_dict: Dict[str, Any]):
        if self.is_normalize_points and self.action_points_num > 0:
            X_U_Pts_ref = self.generate_ref_X_and_U(action,action_obs_dict,k=self.k,s=5.0)
            if X_U_Pts_ref is not None:
                # xref: Entire reference trajectory from x0 to xf as numpy array of size (Kf+1, 3), where each row is (xref [m], yref [m], theta_ref[rad])
                # uref: Entire reference input from ur0 to urf as numpy array of size (Kf, 2), where each row is (vref [m/s], ang_vel_ref [rad/s])
                x_ref,u_ref,action_points_robot_frame = X_U_Pts_ref
                self.mpc.set_ref_trajectory(x_ref, u_ref)
                if self.mpc_mapframe_test_traj:
                    rx_w = self._robot_pose.x
                    ry_w = self._robot_pose.y
                    rtheta_w = self._robot_pose.theta
                    start_pos = np.array([rx_w,ry_w,rtheta_w])
                    status, u = self.mpc.update(start_pos)
                    u = u.flatten()
                    v = u[0]
                    w = u[1]
                    # print("mpc output",[v,0.0,w])
                    return np.array([v,0.0,w], dtype=np.float32), action_points_robot_frame 
                else:
                    start_pos = np.array([0.,0.,0.])
                    status, u = self.mpc.update(start_pos)
                    u = u.flatten()
                    v = u[0]
                    w = u[1]
                    # print("mpc output",[v,0.0,w])
                    return np.array([v,0.0,w], dtype=np.float32), action_points_robot_frame
            else:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.zeros((self.action_points_num, 2), dtype=np.float32)
        else:
            raise ValueError("Convex MPC action is not supported in non-convex MPC encoder.")
        
    def generate_ref_X_and_U(self,action,action_obs_dict: Dict[str, Any], k=2, s=5.0):

        if self.action_points_num > 10:
            self.action_points_num = 10
            rospy.logerr("action_points_num should be less than 10.")
        if action is None:
            rospy.logerr("action is None.")
            return None
        # detect action all > 0 and < 1
        if np.any(action < 0) or np.any(action > 1):
            rospy.logerr("action should be in [0,1].")
            return None

        action_points = []
        action_points_robot = []
        netout_scale_factors = self._action_space_manager.decode_action(action)
        # angle dis
        netout_scale_factors = np.clip(netout_scale_factors, 0.0, 1.0)
        # netout_scale_factors = np.random.rand(2*self.action_points_num)
        # print("netout_scale_factors:",netout_scale_factors)

        if action_obs_dict is not None:
            self.obs_dict_d86 = self.get_observations_d86(action_obs_dict)
            if self.obs_dict_d86["laser_convex"][0] is not None:

                # rvx,rvy = self.worldva2robotva(self.obs_dict_d86["robot_world_vel"].x,self.obs_dict_d86["robot_world_vel"].y,self.obs_dict_d86["robot_world_pose"].theta)
                # # if rvx or rvy is None or nan
                # if rvx is None or rvy is None or np.isnan(rvx) or np.isnan(rvy):
                #     rospy.logerr("rvx or rvy is None or nan.")
                #     rvx = 0.0
                #     rvy = 0.0
                # theta = np.arctan2(rvy, rvx)  # 第一个动作点的方向
                
                # connect robot present pose to the first action point
                if self.if_add_robot_pt:
                    action_points.append((0.,0.))
                    # print("add robot point to action points.")

                if self.goal_in_convex_if_angle:
                    # if goal in convex, the angle of the first action point set to goal
                    goal_robot_frame = action_obs_dict["goal_location_in_robot_frame"]
                    convex_region_robot_frame = self.obs_dict_d86["laser_convex"][0]
                    if self.is_in_convex(goal_robot_frame,convex_region_robot_frame):
                        goal_theta = np.arctan2(goal_robot_frame[1],goal_robot_frame[0])
                        # convert goal_theta to 0-2pi with numpy
                        goal_theta = np.where(goal_theta < 0, goal_theta + 2 * np.pi, goal_theta)
                        goal_theta_scale = goal_theta/(2*np.pi) # goal theta scale 0-1
                        # print("goal_in_convex, goal_theta:",goal_theta,"goal_theta_scale:",goal_theta_scale)
                        for i in range(len(netout_scale_factors)):
                            if i == 0:
                                netout_scale_factors[i] = goal_theta_scale
                            elif i % 2 == 0:
                                netout_scale_factors[i] = 0.0
                            # elif i % 2 == 1:
                            #     netout_scale_factors[i] = 1.0
                
                if len(netout_scale_factors) >= 2:
                    one_action_point = self.calc_polar_action_points(
                        (0.0, 0.0, 0.0),
                        (netout_scale_factors[0], netout_scale_factors[1]),
                        0
                    )
                    action_points.append(one_action_point)
                    action_points_robot.append(one_action_point)

                # 对于后续的动作点，基于前一个动作点的位置和方向进行计算
                for i in range(1, self.action_points_num):
                    sf_index = i * 2
                    if (sf_index + 1) < len(netout_scale_factors):
                        one_action_point = self.calc_polar_action_points(
                            (action_points[-1][0],action_points[-1][1],np.arctan2(action_points[-1][1],action_points[-1][0])),  # 使用前一个动作点的位置和方向
                            (netout_scale_factors[sf_index], netout_scale_factors[sf_index + 1]),
                            i  # 传递feasible_spaces索引
                        )
                        action_points.append(one_action_point)
                        action_points_robot.append(one_action_point)
                
                if not self.goal_in_convex_if_angle:
                    # if goal in convex, the last action point set to goal
                    goal_robot_frame = action_obs_dict["goal_location_in_robot_frame"]
                    convex_region_robot_frame = self.obs_dict_d86["laser_convex"][0]
                    if self.is_in_convex(goal_robot_frame,convex_region_robot_frame):
                        # action_points[-1] = (goal_robot_frame[0],goal_robot_frame[1])
                        action_points.pop()
                        action_points.append(goal_robot_frame)

                
                action_points = np.array(action_points, dtype=np.float32)
                action_points_robot = np.array(action_points_robot, dtype=np.float32) # 保存机器人坐标系下的动作点

                if self.mpc_mapframe_test_traj:
                    # # replace action_points with mpc_xref_traj
                    # self.publish_marker(self.mpc_xref_traj, Marker.LINE_STRIP, 0.0, 1.0, 0.0, "mpc_xref_traj", 0.2 , 0.8)
                    # action_points = self.update_action_points()

                    # convert action_points to world frame
                    action_points_world = []
                    for pt in action_points:
                        pt_w = self.robotpt2worldpt((pt[0],pt[1]))
                        action_points_world.append(pt_w)
                    action_points_world = np.array(action_points_world)
                    action_points = action_points_world
                
                # if nan or none in action_points
                if np.isnan(action_points).any() or np.any(action_points == None):
                    rospy.logerr("action_points has nan or none.")
                    if self._last_action_points is not None:
                        action_points = self._last_action_points
                        rospy.logwarn("action_points has nan or none. Use last action_points.")
                    else:
                        return None
                self._last_action_points = action_points
                action_points = self.add_perturbation_to_duplicates(action_points)

                # # 打印action_points的形状和内容来进行检查
                # print("netout_scale_factors:", netout_scale_factors)
                # print("action_points content:", action_points)

                if self.enable_rviz:
                    self.publish_points_rviz(action_points)

                if self.enable_rviz and self.debug_vis:
                    feasible_space_points = []
                    feasible_spaces = self.obs_dict_d86["feasible_spaces"]
                    for idx, space in enumerate(feasible_spaces):
                        if len(space) > 1:  # 确保space中至少有两个点可以连接
                            # 为每个feasible_space创建一个唯一的namespace
                            namespace = f'feasible_space_{idx}'
                            for i in range(len(space)):
                                p = (space[i] * np.cos(self.obs_dict_d86["laser_convex"][2][i]), space[i] * np.sin(self.obs_dict_d86["laser_convex"][2][i]))
                                # 将space中的点转换为期望的格式
                                feasible_space_points.append(p)
                            # 发布连线
                            self.publish_marker(feasible_space_points, Marker.POINTS, 1.0, 1.0 - idx * 0.1, 0.0, namespace, 0.05 , 0.8)

                # # 计算每个点到原点(0, 0)的欧氏距离
                # distances = np.linalg.norm(action_points, axis=1)
                # # 找出所有距离大于或等于8的点的索引
                # indices = np.where(distances >= 2.2)[0]
                # if len(indices) > 0:
                #     print("以下action_points点离原点(0,0)的距离大于或等于8：")
                #     for idx in indices:
                #         print(f"点{idx + 1}: {action_points[idx]}，距离: {distances[idx]:.2f}")
                #     print("============================================")
                #     # save obs_dict file
                #     np.save("/home/dmz/Documents/obs_dict.npy",self.obs_dict_d86)
                #     initial_pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)

                # 确保点的数量足够
                if action_points.shape[0] < k + 1:
                    raise ValueError("动作点的数量不足以进行样条插值")

                # 确保点的维度正确（这里我们假设是二维点，即n=2）
                if action_points.ndim != 2 or action_points.shape[1] != 2:
                    raise ValueError("点的格式应为(m, 2)，其中m是点的数量")
                
                # 检查数据点是否包含NaN或无穷值
                if np.isnan(action_points).any() or np.isinf(action_points).any():
                    print("数据点包含NaN或无穷值: ", action_points)
                    raise ValueError("数据点包含NaN或无穷值")

                try:
                    '''
                    action_points.T: 这里 action_points 应该是一个形状为 (N, 2) 的数组，其中 N 表示点的数量，2 表示每个点的 x 和 y 坐标。.T 是 NumPy 数组的转置操作，用于将数组的形状从 (N, 2) 转置为 (2, N)，满足 splprep 输入参数的要求，即每一行代表一个维度。
                    s=s: 平滑因子 s 控制样条曲线的平滑程度。较小的 s 值会使得曲线尽可能地经过所有给定的点（即更少的平滑），而较大的 s 值会使曲线更加平滑，即使它不完全经过所有点。如果 s 设置为 0，样条曲线会经过所有的点（插值条件）。
                    per=False: 指定生成的样条曲线是否为周期性的。如果设置为 True，那么样条曲线的开始和结束将在空间中形成闭环。
                    k=2: 样条的次数。这个参数决定了样条曲线的平滑程度和弯曲能力。k=2 表示使用二次样条。
                        如果数据点之间的关系接近直线，使用线性样条（k=1）可能更合适。
                        如果需要一定的曲线弯曲能力但又不希望太复杂，可以选择二次样条（k=2）。
                        对于需要高度平滑曲线的应用，如道路或轨迹设计，三次样条（k=3）是常用的选择
                    der=0 参数表示我们要计算的是曲线上的点，而不是其导数。
                    '''
                    dt = self._step_size
                    tck, u = splprep(action_points.T, s=s, per=False, k=k)
                    u_new = np.linspace(u.min(), u.max(), 10)
                    x_new, y_new = splev(u_new, tck, der=0)
                    dx_new, dy_new = splev(u_new, tck, der=1)  # 获取曲线上点的一阶导数（即切线）
                    # 计算每个点处的切线角度
                    theta_new_mod = np.arctan2(dy_new, dx_new)
                    # 构建xref
                    Xref = np.vstack((x_new, y_new, theta_new_mod)).T
                    # 计算速度和角速度作为uref
                    v = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2) / dt
                    w = np.diff(theta_new_mod) / dt
                    # normalize w to [-pi,pi] 模运算
                    w = (w + np.pi) % (2 * np.pi) - np.pi
                    # 构建uref
                    Uref = np.vstack((v, w)).T

                    if self.enable_rviz:
                        self.publish_spline_rviz(x_new, y_new)

                    # # 合并 x_new 和 y_new 为一个二维数组
                    # spline_points = np.vstack((x_new, y_new)).T  # T 是转置操作，将行向量转换为列向量
                    # # 计算每个点到原点(0, 0)的欧氏距离
                    # distances = np.linalg.norm(spline_points, axis=1)
                    # # 找出所有距离大于或等于8的点的索引
                    # indices = np.where(distances >= 8)[0]
                    # if len(indices) > 0:
                    #     print("以下spline曲线上的点离原点(0,0)的距离大于或等于8：")
                    #     time.sleep(5)
                    #     for idx in indices:
                    #         print(f"点{idx + 1}: {spline_points[idx]}，距离: {distances[idx]:.2f}")
                    return Xref, Uref, action_points_robot
                except Exception as e:
                    print("b-spline拟合过程中发生错误:", e)
                    rospy.logerr("b-spline拟合过程中发生错误")
                    return None
        return None
        
    def add_perturbation_to_duplicates(self, action_points):
        """
        Adds perturbation to duplicate points in the action points.
        To avoid duplicate points in the action points
        """
        perturbed_points = []
        seen = {}  # 用于记录每个点出现的次数
        for point in action_points:
            point_tuple = tuple(point)
            if point_tuple in seen and seen[point_tuple] > 0:
                # 如果这个点已经出现过，对当前点添加扰动
                perturbed_point = point + np.random.normal(0, 1e-3, size=point.shape)
                perturbed_points.append(perturbed_point)
                seen[point_tuple] += 1
            else:
                perturbed_points.append(point)
                seen[point_tuple] = 1
        return np.array(perturbed_points, dtype=np.float32)

    def get_observations_d86(self, action_obs_dict: Dict[str, Any]):
        msg_galaxy2d: Galaxy2D= action_obs_dict["laser_convex"]
        state: Odometry = action_obs_dict["robot_state"]

        robot_state= self.process_robot_state_msg(state)

        self._robot_pose = robot_state[0]
        self._robot_vel = robot_state[1]

        g2d_cal_success = msg_galaxy2d.success.data

        if msg_galaxy2d.scans is None or len(msg_galaxy2d.scans) != self.laser_num_beams:
            print(f"Galaxy2D scan is None or not complete {self.laser_num_beams} degrees.")
            rospy.logwarn(f"Galaxy2D scan is None or not complete {self.laser_num_beams} degrees.")
            scans = [0. for i in range(self.laser_num_beams)]
        else:
            scans = msg_galaxy2d.scans

        g2d_convex_vertex = []
        for p in msg_galaxy2d.convex_vertex.points:
            g2d_convex_vertex.append((p.x,p.y))

        g2d_polar_convex = msg_galaxy2d.polar_convex
        g2d_polar_convex_theta = msg_galaxy2d.polar_convex_theta

        # local_feasible_space,global_feasible_space = \
        # self.feasible_position_d86(scans,g2d_cal_success,g2d_polar_convex)

        feasible_spaces = self.feasible_position(scans, g2d_cal_success, g2d_polar_convex)

        if g2d_cal_success and len(g2d_polar_convex) == self._max_vertex_num:
            process_scan = []
        else:
            print("Fail to get g2d_polar_convex ,so get convex_vertex from scans.")
            rospy.logwarn("Fail to get g2d_polar_convex ,so get convex_vertex from scans.")
            g2d_cal_success = False
            g2d_convex_vertex = []
            g2d_polar_convex = []
            g2d_polar_convex_theta = [i*2*np.pi/self._max_vertex_num for i in range(self._max_vertex_num)]
            # 0-359
            di = int(self.laser_num_beams/self._max_vertex_num)
            for i in range(self._max_vertex_num):
                g2d_convex_vertex.append((scans[di*i]*np.cos(g2d_polar_convex_theta[i]),scans[di*i]*np.sin(g2d_polar_convex_theta[i])))
                g2d_polar_convex.append(scans[di*i])

        obs_dict_d86 = {
            "laser_scan": np.array(scans),
            "robot_world_pose": self._robot_pose,
            "robot_world_vel": self._robot_vel.linear,
            "feasible_spaces":feasible_spaces,
            "laser_convex":[g2d_convex_vertex,g2d_polar_convex,g2d_polar_convex_theta,g2d_cal_success],
        }
        

        return obs_dict_d86

    def feasible_position(self, scans, g2d_cal_success, g2d_polar_convex):
        # 初始化可行空间列表
        feasible_spaces = []
        
        # 最大线速度
        max_linear_speed = 0.7  # m/s

        speed_factor = self._feasible_position_speed_factor

        # 根据self.action_points_num计算时间间隔的索引
        time_intervals = np.linspace(1, 10, self.action_points_num, dtype=int)

        for index in time_intervals:
            # 计算每个时间间隔的可行距离
            time_step = index * self._step_size
            feasible_dist = max_linear_speed* speed_factor * time_step

            # 初始化当前时间间隔的可行空间
            current_feasible_space = []
            
            if g2d_cal_success:
                for dist in g2d_polar_convex:
                    current_feasible_space.append(min(dist, feasible_dist))
            else:
                for dist in scans:
                    current_feasible_space.append(min(dist, feasible_dist))

            # 将当前时间间隔的可行空间添加到总列表中
            feasible_spaces.append(current_feasible_space)

        return feasible_spaces


    def feasible_position_d86(self,scans,g2d_cal_success,g2d_polar_convex):
        '''
        scans: 一个数组，包含激光雷达扫描的距离数据，表示机器人周围环境的几何形状。
        g2d_cal_success: 一个布尔值，表示Galaxy2D计算是否成功。
        g2d_polar_convex: 一个数组，包含从Galaxy2D计算得到的极坐标下的凸包顶点。
        '''
        local_feasible_dist = -10.
        global_feasible_dist = 10.
        # global_feasible_dist = -10.
        # local_feasible_dist = 10.
        plan_dynamic_limit:tuple=(2.5,2.,4.)
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


    def calc_polar_action_points(self,start,action,whcich):
        """
        """
        # refer polar action point angle
        # 基于动作参数和起始点的朝向（start[2]），计算目标动作点相对于起始点的极坐标角度。
        theta = self.NormalizeAngleTo2Pi(2*np.pi*action[0] + start[2])
        thetas = []
        # # debug
        pts_world = []
        convex_scans_limit_range = self.obs_dict_d86["feasible_spaces"][whcich]
        limit_range_from_start = []
        
        # obs_dict_d86["laser_convex"]: [g2d_convex_vertex, g2d_polar_convex, g2d_polar_convex_theta, g2d_cal_success]
        for i in range(self._max_vertex_num):
            # 对于每个给定点p（points in feasible_space），首先计算该点相对于起始点start的位置向量的角度。
            p = (convex_scans_limit_range[i] * np.cos(self.obs_dict_d86["laser_convex"][2][i]), \
                convex_scans_limit_range[i] * np.sin(self.obs_dict_d86["laser_convex"][2][i]))
            # add dis from start to p
            limit_range_from_start.append(np.hypot(p[0] - start[0], p[1] - start[1]))

            # # # debug
            # p_world = self.robotpt2worldpt(p)
            # pts_world.append(p_world)

            thetas.append(self.NormalizeAngleTo2Pi(np.arctan2(p[1] - start[1], p[0] - start[0])))
        
        # print("---------------------------")
        # print("robot_pose:",self.robotpt2worldpt((0,0)))
        # print("theta:",theta)
        # for i in range(self._max_vertex_num):
        #     print("i:",i," thetas:",thetas[i]," convex_scans_limit_range:",convex_scans_limit_range[i]," pts_world:",pts_world[i])


        # # 在一个顺序排列的角度列表（thetas）中找到一个区间，使得给定的角度（theta）位于这个区间的两个index（prev和next）之间。
        # _find = False
        # next = min_index
        # prev = min_index-1
        # for j in range(min_index+1):
        #     if theta < thetas[j]:
        #         next = j
        #         prev = next - 1
        #         _find = True
        #         break
        # if not _find:
        #     for j in range(min_index,self._max_vertex_num):
        #         if theta < thetas[j]:
        #             next = j
        #             prev = next - 1
        #             break
        
        # 找到最接近且小于等于 theta 的值（prev），以及最接近且大于 theta 的值（next）。
        prev, next = None, None
        prev_diff, next_diff = float('inf'), float('inf')

        for i, val in enumerate(thetas):
            diff = theta - val
            if diff == 0:
                prev, next = i, i
                break
            elif diff > 0 and diff < prev_diff:
                prev_diff = diff
                prev = i
            elif diff < 0 and -diff < next_diff:
                next_diff = -diff
                next = i

        # 处理找不到 prev 或 next 的情况（例如，当 theta 大于所有值或小于所有值时）
        if prev is None and next is None:
            rospy.logerr("prev and next are all None,return None")
            return None
        elif prev is None: prev = next - 1
        elif next is None: next = (prev + 1) % len(thetas)
        elif prev == next: next = (prev + 1) % len(thetas)

        # start: 机器人的当前位置坐标(x, y)。
        # 第二个参数是一个由start点和theta角定义的向量的终点坐标。
        # 这里，self._laser_max_range表示这个向量的长度，乘以np.cos(theta)和np.sin(theta)计算出向量在x和y方向上的分量，确保这个向量沿着theta角延伸。
        # 第三和第四个参数分别是机器人可行空间边界上的两个顶点prev和next，
        p11 = (start[0], start[1])
        p12 = (start[0] + self._laser_max_range * np.cos(theta), start[1] + self._laser_max_range * np.sin(theta))
        p21 = (start[0] + limit_range_from_start[prev] * np.cos(thetas[prev]), start[1] + limit_range_from_start[prev] * np.sin(thetas[prev]))
        p22 = (start[0] + limit_range_from_start[next] * np.cos(thetas[next]), start[1] + limit_range_from_start[next] * np.sin(thetas[next]))
        crossp = self.calc_cross_point(p11,p12,p21,p22)

        # p11_world = self.robotpt2worldpt(p11)
        # p12_world = self.robotpt2worldpt(p12)
        # p21_world = self.robotpt2worldpt(p21)
        # p22_world = self.robotpt2worldpt(p22)
        # crossp_world = self.robotpt2worldpt(crossp)
        # print(f"p11_world = {p11_world}, p12_world = {p12_world}")
        # print(f"p21_world = {p21_world}, p22_world = {p22_world}")
        # print(f"crossp_world = {crossp_world}")
        
        if self.enable_rviz and self.debug_vis:
            # 可视化从start点出发的线段
            line_start = p11
            line_end = p12
            self.publish_marker([line_start, line_end], Marker.LINE_STRIP, 0.0, 1.0, 0.0, 'start_line', 0.1, 0.4)

            # 可视化连接prev和next的线段
            line_prev = p21
            line_next = p22
            self.publish_marker([line_prev, line_next], Marker.LINE_STRIP, 0.0, 0.5, 1.0, 'prev_next_line', 0.1, 1.0)

        # refer polar action point distance
        dist = np.hypot(crossp[0] - start[0], crossp[1] - start[1]) * action[1]


        action_point = (start[0] + dist * np.cos(theta), start[1] + dist * np.sin(theta))
        # polar_point = (np.arctan2(action_point[1],action_point[0]),np.hypot(action_point[1],action_point[0]))

        return  action_point
    

    def update_action_points(self):

        # 重新确定选取点的数量，取最小值为10或剩余点的数量
        num_points_to_select_from = min(10, len(self.mpc_xref_traj))
        
        # 如果剩余点不足self.action_points_num个，则选取一个点复制多次
        if len(self.mpc_xref_traj) < self.action_points_num:
            selected_indices = [0] * self.action_points_num
        else:
            # 计算要选取的点的索引
            selected_indices = np.linspace(0, num_points_to_select_from - 1, num=self.action_points_num, dtype=int)
        
        # 从mpc_xref_traj中选取对应的点
        action_points = self.mpc_xref_traj[selected_indices]

        rx_world, ry_world = self._robot_pose.x, self._robot_pose.y
        
        # 查找离机器人最近的点
        distances = np.sqrt((self.mpc_xref_traj[:,0] - rx_world)**2 + (self.mpc_xref_traj[:,1] - ry_world)**2)
        nearest_point_index = np.argmin(distances)
        nearest_point_distance = distances[nearest_point_index]

        # 如果最近的点离机器人的距离小于0.3，去掉该点及其之前的所有点
        if nearest_point_distance < 0.25:
            self.mpc_xref_traj = self.mpc_xref_traj[nearest_point_index + 1:]
        
        # add robot pose to the first of action_points
        action_points = np.insert(action_points, 0, [rx_world, ry_world], axis=0)
        
        return action_points

    
    def NormalizeAngleTo2Pi(self, d_theta):# 0-2pi
        d_theta_normalize = d_theta
        while d_theta_normalize > 2 * math.pi:
            d_theta_normalize = d_theta_normalize - 2 * math.pi
        while d_theta_normalize < 0:
            d_theta_normalize = d_theta_normalize + 2 * math.pi
        return d_theta_normalize
    
    @staticmethod
    def calc_cross_point(a1,a2,b1,b2):
        '''
        计算两条线段的交点
        a1, a2：第一条线段的起点和终点坐标，分别表示为(x, y)形式的元组。
        b1, b2：第二条线段的起点和终点坐标，同样表示为(x, y)形式的元组
        return：交点坐标，表示为(x, y)形式的元组
        '''
        def cross(a,b):
            return a[0]*b[1]-a[1]*b[0] 
        a = (a2[0]-a1[0],a2[1]-a1[1])
        b = (b2[0]-b1[0],b2[1]-b1[1])
        frac = cross((b1[0]-a1[0],b1[1]-a1[1]),b)
        num = cross(a,b)
        t = frac/num
        return (a1[0]+t*a[0],a1[1]+t*a[1])
    
    def convert_to_point(self, coords):
        """将坐标转换为ROS Point消息。"""
        x, y = coords
        return Point(x=x, y=y, z=0)

    def robotpt2worldpt(self, point_robot_frame):
        """
        将点从机器人坐标系转换到全局坐标系。

        参数:
        - point_robot_frame: 机器人坐标系中的点，格式为(x, y)元组。

        返回:
        - 转换后的点在全局坐标系中的位置，格式为(x, y)元组。
        """
        # 从 obs_dict_d86 字典中提取机器人的全局位置和朝向
        robot_pose = self._robot_pose # type Pose2d
        robot_x, robot_y, robot_theta = robot_pose.x, robot_pose.y, robot_pose.theta
        
        # 机器人坐标系中的点
        point_x, point_y = point_robot_frame
        
        # 计算点在全局坐标系中的位置
        world_x = robot_x + (point_x * np.cos(robot_theta) - point_y * np.sin(robot_theta))
        world_y = robot_y + (point_x * np.sin(robot_theta) + point_y * np.cos(robot_theta))
        
        return (world_x, world_y)
    
    @staticmethod
    def worldva2robotva(wvax,wvay,now_wyaw):
        va =np.hypot(wvax,wvay)
        vayaw = np.arctan2(wvay,wvax)
        vax = np.cos(vayaw - now_wyaw)*va
        vay = np.sin(vayaw - now_wyaw)*va
        return vax,vay

    @staticmethod
    def worldp2robotp(point_in_world,robot_pose_in_world):
        wx,wy = point_in_world
        dist = np.hypot(wx - robot_pose_in_world[0],wy - robot_pose_in_world[1])
        p_theta_world = np.arctan2(wy - robot_pose_in_world[1],wx - robot_pose_in_world[0])
        return  dist*np.cos(p_theta_world - robot_pose_in_world[2]) ,dist*np.sin(p_theta_world - robot_pose_in_world[2])

    @staticmethod
    def robotp2worldp(point_in_robort,robot_pose_in_world):
        rx,ry= point_in_robort
        dist = np.hypot(rx,ry)
        p_theta_world = robot_pose_in_world[2] + np.arctan2(ry,rx)
        return robot_pose_in_world[0] + dist*np.cos(p_theta_world) ,robot_pose_in_world[1] + dist*np.sin(p_theta_world)

    @staticmethod
    def robotva2worldva(rvax,rvay,now_wyaw):
        va =np.hypot(rvax,rvay)
        vayaw = np.arctan2(rvay,rvax)
        vax = np.cos(vayaw + now_wyaw)*va
        vay = np.sin(vayaw + now_wyaw)*va
        return vax,vay
    
    @staticmethod
    def is_in_convex(point,convex):
        # convex为顺时针
        for i in range(1,len(convex)):
            cross = convex[i-1][0]*convex[i][1] - convex[i-1][1]*convex[i][0] +  (convex[i][0] - convex[i-1][0])*point[1]+ (convex[i-1][1] - convex[i][1])*point[0]
            if cross > 0:
                return False
        return True

    
    def publish_spline_rviz(self, x_new, y_new):
        # 曲线可视化
        marker_line = Marker()
        if self.mpc_mapframe_test_traj:
            marker_line.header.frame_id = "map"
        else:
            marker_line.header.frame_id = "eval_sim/robot"
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.1  # 线宽
        marker_line.color.a = 1.0  # 不透明度
        marker_line.color.r = 0.0
        marker_line.color.g = 0.0
        marker_line.color.b = 1.0  # 蓝色曲线

        for x, y in zip(x_new, y_new):
            p = Point()
            p.x = x
            p.y = y
            p.z = 0
            marker_line.points.append(p)
        self.spline_pub.publish(marker_line)

    def publish_points_rviz(self, action_points):
        # 动作点可视化
        marker_points = Marker()
        if self.mpc_mapframe_test_traj:
            marker_points.header.frame_id = "map"
        else:
            marker_points.header.frame_id = "eval_sim/robot"
        marker_points.type = Marker.POINTS
        marker_points.action = Marker.ADD
        marker_points.scale.x = 0.15 # 点大小
        marker_points.scale.y = 0.15
        marker_points.color.a = 1.0  # 不透明度
        marker_points.color.r = 1.0  # 红色点
        marker_points.color.g = 0.0
        marker_points.color.b = 0.0

        for point in action_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0
            marker_points.points.append(p)
        self.action_points_pub.publish(marker_points)

    def publish_marker(self, points, marker_type, r, g, b, namespace='visualization', scale=0.1, trans = 1.0):
        marker = Marker()
        marker.header.frame_id = "eval_sim/robot"
        marker.ns = namespace
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color.a = trans
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        """
        LINE_STRIP将一系列点连接成一串连续的线段。如果你有点A、B、C，则LINE_STRIP会绘制线段AB和线段BC。每个点（除了第一个点）都会与前一个点连接起来，形成一条连续的线路。这种方式适合绘制路径或者连续的线条。
        示例: 给定点序列 [A, B, C, D]，LINE_STRIP会绘制线段 AB, BC, CD。
        LINE_LIST将点对视为独立的线段。这意味着每两个点形成一条线段，点之间不共享。如果你有点A、B、C、D，则LINE_LIST会绘制线段AB和线段CD。点是成对解释的，每对点定义一条线段。
        示例: 给定点序列 [A, B, C, D]，LINE_LIST会绘制线段 AB, CD。
        """

        if marker_type == Marker.LINE_STRIP or marker_type == Marker.LINE_LIST:
            # 对于线段，points是线段的端点列表
            marker.points = [Point(x=p[0], y=p[1], z=0) for p in points]
        elif marker_type == Marker.POINTS:
            # 对于点，points是点的列表
            marker.points = [Point(x=p[0], y=p[1], z=0) for p in points]
            marker.scale.x = 0.06  # 点的大小
            marker.scale.y = 0.06


        self.marker_pub.publish(marker)

