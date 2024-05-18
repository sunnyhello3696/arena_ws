"""
This file is used to calculate from the simulation data, various metrics, such as
- did a collision occur
- how long did the robot take form start to goal
the metrics / evaluation data will be saved to be preproccesed in the next step
"""
import enum
import typing
from typing import List
import numpy as np
import pandas as pd
import os
from pandas.core.api import DataFrame as DataFrame
import yaml
import rospkg
import json

from arena_evaluation.utils import Utils

class Action(str, enum.Enum):
    STOP = "STOP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"


class DoneReason(str, enum.Enum):
    TIMEOUT = "TIMEOUT"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"


class Metric(typing.TypedDict):

    time: typing.List[int]
    time_diff: int
    episode: int
    goal: typing.List
    start: typing.List

    path: typing.List
    path_length_values: typing.List
    path_length: float
    angle_over_length: float
    curvature: typing.List
    normalized_curvature: typing.List 
    roughness: typing.List

    cmd_vel: typing.List
    velocity: typing.List
    acceleration: typing.List
    jerk: typing.List
    
    collision_amount: int
    collisions: typing.List
    
    action_type: typing.List[Action]
    result: DoneReason

class PedsimMetric(Metric, typing.TypedDict):

    num_pedestrians: int

    avg_velocity_in_personal_space: float
    total_time_in_personal_space: int
    time_in_personal_space: typing.List[int]

    total_time_looking_at_pedestrians: int
    time_looking_at_pedestrians: typing.List[int]

    total_time_looked_at_by_pedestrians: int
    time_looked_at_by_pedestrians: typing.List[int]


class Config:
    TIMEOUT_TRESHOLD = 75
    MAX_COLLISIONS = 1
    MIN_EPISODE_LENGTH = 5
    
    PERSONAL_SPACE_RADIUS = 1 # personal space is estimated at around 1'-4'
    ROBOT_GAZE_ANGLE = np.radians(5) # size of (one half of) direct robot gaze cone
    PEDESTRIAN_GAZE_ANGLE = np.radians(5) # size of (one half of) direct ped gaze cone
    # 表示机器人或行人注视方向的角度范围
    # np.radians() 是 NumPy 库中的一个函数，用于将角度值转化为弧度值
class Math:

    @classmethod
    def round_values(cls, values, digits=3):
        # 保留有效数字
        return [round(v, digits) for v in values]

    @classmethod
    def grouping(cls, base: np.ndarray, size: int) -> np.ndarray:
        return np.moveaxis( 
            np.array([
                np.roll(base, i, 0)
                for i
                in range(size)
            ]),
            [1],
            [0]
        )[:-size]

    @classmethod
    def triangles(cls, position: np.ndarray) -> np.ndarray:
        return cls.grouping(position, 3)
    
    @classmethod
    def triangle_area(cls, vertices: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            np.cross(
                vertices[:,1] - vertices[:,0],
                vertices[:,2] - vertices[:,0],
                axis=1
            ),
            axis=1
        ) / 2
    
    @classmethod
    def path_length(cls, position: np.ndarray) -> np.ndarray:
        """
        首先，调用 grouping 方法，将路径点分组成两两一组，形成路径中相邻的点的对。
        然后，计算每对相邻点之间的距离，通过计算每个对中两个点之间的欧氏距离。
        最后，将这些距离组合成一个NumPy数组，并返回该数组作为路径长度。
        """
        pairs = cls.grouping(position, 2)
        return np.linalg.norm(pairs[:,0,:] - pairs[:,1,:], axis=1)

    @classmethod
    def curvature(cls, position: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the curvature of a path. The curvature is defined as the inverse of the radius of the circle that best fits the path at a given point.
        计算路径的曲率和归一化路径曲率。曲率定义为在给定点最适合路径的圆的半径的倒数。
        curvature：一个NumPy数组，其中包含了路径的曲率数据。与输入中的路径点一一对应。
        normalized：一个NumPy数组，其中包含了路径的归一化曲率数据。同样，与输入中的路径点一一对应。
        """

        triangles = cls.triangles(position)

        d01 = np.linalg.norm(triangles[:,0,:] - triangles[:,1,:], axis=1)
        d12 = np.linalg.norm(triangles[:,1,:] - triangles[:,2,:], axis=1)
        d20 = np.linalg.norm(triangles[:,2,:] - triangles[:,0,:], axis=1)

        triangle_area = cls.triangle_area(triangles)
        divisor = np.prod([d01, d12, d20], axis=0)
        divisor[divisor==0] = np.nan

        curvature = 4 * triangle_area / divisor
        curvature[np.isnan(divisor)] = 0

        normalized = np.multiply(
            curvature,
            d01 + d12
        )

        return curvature, normalized

    @classmethod
    def roughness(cls, position: np.ndarray) -> np.ndarray:
        """
        计算路径的粗糙度。
            首先，根据输入的路径位置数据，将路径划分成相邻的三角形。
            然后，计算每个三角形的面积。
            接着，计算每个三角形的一条边的长度，通常取相邻两个顶点之间的距离。
            最后，根据公式，使用三角形的面积和边长来计算路径的粗糙度。
        粗糙度指标可以帮助评估路径的平滑程度。如果路径的粗糙度较高，表示路径中存在较多的起伏和曲折，可能需要更多的调整和控制来保持稳定的导航。
        """
        
        triangles = cls.triangles(position)

        triangle_area = cls.triangle_area(triangles)
        length = np.linalg.norm(triangles[:,:,0] - triangles[:,:,2], axis=1)
        length[length==0] = np.nan

        roughness = 2 * triangle_area / np.square(length)
        roughness[np.isnan(length)] = 0

        return roughness

    @classmethod
    def acceleration(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(speed)

    @classmethod
    def jerk(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(np.diff(speed))

    @classmethod
    def turn(cls, yaw: np.ndarray) -> np.ndarray:
        pairs = cls.grouping(yaw, 2)
        return cls.angle_difference(pairs[:,0], pairs[:,1])
    
    @classmethod
    def angle_difference(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            return np.pi - np.abs(np.abs(x1 - x2) - np.pi)

class Metrics:

    dir: str
    _episode_data: typing.Dict[int, Metric]

    def _load_data(self) -> typing.List[pd.DataFrame]:

        # converters 参数允许你传递一个字典，其中键是列名，值是一个函数，用于对该列的值进行转换。
        # 定义了一个转换器函数，它将名为 "data" 的列中的值转换为整数类型。如果值的长度小于等于 0，则转换器函数将值设置为 0。
        episode = pd.read_csv(os.path.join(self.dir, "episode.csv"), converters={
            "data": lambda val: 0 if len(val) <= 0 else int(val) 
        })

        # rename(columns={"data": "laserscan"}) 用于重命名 DataFrame 中的列。在第二个例子中，我们将列名为 "data" 的列重命名为 "laserscan"。
        laserscan = pd.read_csv(os.path.join(self.dir, "scan.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "laserscan"})

        # 这个 lambda 函数首先将字符串中的单引号替换为双引号，然后使用 json.loads 函数将其解析为 Python 对象。
        odom = pd.read_csv(os.path.join(self.dir, "odom.csv"), converters={
            "data": lambda col: json.loads(col.replace("'", "\""))
        }).rename(columns={"data": "odom"})

        # string_to_float_list接受一个格式为 "[1.0, 2.0, 3.0]" 的字符串，然后将其转换为包含浮点数的列表 [1.0, 2.0, 3.0]
        cmd_vel = pd.read_csv(os.path.join(self.dir, "cmd_vel.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "cmd_vel"})

        start_goal = pd.read_csv(os.path.join(self.dir, "start_goal.csv"), converters={
            "start": Utils.string_to_float_list,
            "goal": Utils.string_to_float_list
        })

        return [episode, laserscan, odom, cmd_vel, start_goal]

    def __init__(self, dir: str):

        self.dir = dir
        # self.robot_params = self._get_robot_params()

        data = pd.concat(self._load_data(), axis=1, join="inner")
        data = data.loc[:,~data.columns.duplicated()].copy()

        i = 0

        episode_data = self._episode_data = {}

        while True:
            current_episode = data[data["episode"] == i]
            
            # check current_episode's data items length
            if len(current_episode) < Config.MIN_EPISODE_LENGTH:
                break

            # Remove the first three time steps from current_episode
            current_episode = current_episode.iloc[3:]
            
            episode_data[i] = self._analyze_episode(current_episode, i)
            i = i + 1

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._episode_data).transpose().set_index("episode")

    
    def _analyze_episode(self, episode: pd.DataFrame, index) -> Metric:

        # 将纳秒转换为秒
        # episode["time"] /= 10**10
        episode.loc[:, "time"] /= 10**10
        
        positions = np.array([frame["position"] for frame in episode["odom"]])
        velocities = np.array([frame["velocity"] for frame in episode["odom"]])

        curvature, normalized_curvature = Math.curvature(positions)
        roughness = Math.roughness(positions)

        vel_absolute = np.linalg.norm(velocities, axis=1)
        acceleration = Math.acceleration(vel_absolute)
        jerk = Math.jerk(vel_absolute)

        collisions, collision_amount = self._get_collisions(
            episode["laserscan"],
            0.177 + 0.01 # turtlebot radius + safety margin
        )

        path_length = Math.path_length(positions)
        turn = Math.turn(positions[:,2])

        time = list(episode["time"])[-1] - list(episode["time"])[0]

        start_position = self._get_mean_position(episode, "start")
        goal_position = self._get_mean_position(episode, "goal")

        # print("PATH LENGTH", path_length, path_length_per_step)

        return Metric(
            # curvature 和 normalized_curvature 是路径的曲率和归一化曲率数据。
            curvature = Math.round_values(curvature),
            normalized_curvature = Math.round_values(normalized_curvature),

            # roughness 是路径的粗糙度数据。
            roughness = Math.round_values(roughness),

            # path_length_values 是positions group by 2的长度array
            path_length_values = Math.round_values(path_length),
            # path_length 是路径的总长度。
            path_length = path_length.sum(),

            # velocity、acceleration 和 jerk 是路径的速度、加速度和加加速度数据。
            acceleration = Math.round_values(acceleration),
            jerk = Math.round_values(jerk),
            velocity = Math.round_values(vel_absolute),

            # collision_amount 是碰撞次数，collisions 是碰撞的时间点index
            collision_amount = collision_amount,
            collisions = list(collisions),

            # path 是路径的位置数据，angle_over_length 是路径的角度和长度之比。
            path = [list(p) for p in positions],
            angle_over_length = np.abs(turn.sum() / path_length.sum()),

            # action_type 是路径的动作类型数据，result 是路径的结果数据。
            action_type = list(self._get_action_type(episode["cmd_vel"])),

            time_diff = time, ## Ros time in s
            time = list(map(int, episode["time"].tolist())),

            episode = index,

            result = self._get_success(time, collision_amount),
            cmd_vel = list(map(list, episode["cmd_vel"].to_list())),
            goal = goal_position,
            start = start_position
        )
    
    def _get_robot_params(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            content = yaml.safe_load(file)

            model = content["model"]

        robot_model_params_file = os.path.join(
            rospkg.RosPack().get_path("arena-simulation-setup"), 
            "robot", 
            model, 
            "model_params.yaml"
        )

        with open(robot_model_params_file, "r") as file:
            return yaml.safe_load(file)

    def _get_mean_position(self, episode, key):
        """
        该函数用于计算给定键对应列的平均位置。
        该函数接受两个参数：episode 和 key。episode 是一个 DataFrame，包含了一个名为 key 的列，该列包含了位置数据。
            然后，它遍历位置列表中的每个位置，并将其转换为字符串格式，并使用冒号连接起来形成一个哈希值。
            每次出现一个新的哈希值时，将其添加到计数器中，并增加计数器的值。
            接着，它对计数器中的项按照键进行排序，以确保具有相同哈希值的位置按照其出现次数排序。
            最后，它从排序后的位置中获取第一个位置的哈希值，并将其转换为浮点数列表作为平均位置返回。
        """
        positions = episode[key].to_list()
        counter = {}

        for p in positions:
            hash = ":".join([str(pos) for pos in p])

            counter[hash] = counter.get(hash, 0) + 1

        sorted_positions = dict(sorted(counter.items(), key=lambda x: x))

        return [float(r) for r in list(sorted_positions.keys())[0].split(":")]

    def _get_position_for_collision(self, collisions, positions):
        for i, collision in enumerate(collisions):
            collisions[i][2] = positions[collision[0]]

        return collisions

    def _get_success(self, time, collisions):
        if time >= Config.TIMEOUT_TRESHOLD:
            return DoneReason.TIMEOUT

        if collisions >= Config.MAX_COLLISIONS:
            return DoneReason.COLLISION

        return DoneReason.GOAL_REACHED
    
    def _get_collisions(self, laser_scans, lower_bound):
        """
        Calculates the collisions. Therefore, 
        the laser scans is examinated and all values below a 
        specific range are marked as collision.

        Argument:
            - Array laser scans representing the scans over
            time
            - the lower bound for which a collisions are counted

        Returns tupel of:
            - Array of tuples with indexs and time in which
            a collision happened
        """
        collisions = []
        collisions_marker = []


        # 函数通过循环遍历 laser_scans 中的每个时间点，并检查该时间点上的激光扫描数据是否有任何值小于或等于 lower_bound。
        # 如果有，则将该时间点标记为发生了碰撞，并将其索引记录在 collisions 列表中。
        for i, scan in enumerate(laser_scans):

            is_collision = len(scan[scan <= lower_bound]) > 0

            collisions_marker.append(is_collision)
            
            if is_collision:
                collisions.append(i)

        # 它遍历 collisions_marker 列表，并检查相邻的标记之间是否由未发生碰撞变为发生了碰撞。如果是，则将碰撞次数加一。
        collision_amount = 0
        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_action_type(self, actions):
        action_type = []

        for action in actions:
            if sum(action) == 0:
                action_type.append(Action.STOP.value)
            elif action[0] == 0 and action[1] == 0:
                action_type.append(Action.ROTATE.value)
            else:
                action_type.append(Action.MOVE.value)

        return action_type

    
        
class PedsimMetrics(Metrics):

    def _load_data(self) -> List[DataFrame]:
        pedsim_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters = {"data": Utils.parse_pedsim}
        ).rename(columns={"data": "peds"})
        
        return super()._load_data() + [pedsim_data]
    
    def __init__(self, dir: str, **kwargs):
        super().__init__(dir=dir, **kwargs)

    def _analyze_episode(self, episode: pd.DataFrame, index):

        super_analysis = super()._analyze_episode(episode, index)

        robot_position = np.array([odom["position"][:2] for odom in episode["odom"]])
        peds_position = np.array([[ped.position for ped in peds] for peds in episode["peds"]])

        # list of (timestamp, ped) indices, duplicate timestamps allowed
        personal_space_frames = np.linalg.norm(peds_position - robot_position[:,None], axis=-1) <= Config.PERSONAL_SPACE_RADIUS
        # list of timestamp indices, no duplicates
        is_personal_space = personal_space_frames.max(axis=1)

        # time in personal space
        time = np.diff(np.array(episode["time"]), prepend=0)
        total_time_in_personal_space = time[is_personal_space].sum()
        time_in_personal_space = [time[frames].sum(axis=0).astype(np.integer) for frames in personal_space_frames.T]

        # v_avg in personal space
        velocity = np.array(super_analysis["velocity"])
        velocity = velocity[is_personal_space]
        avg_velocity_in_personal_space = velocity.mean() if velocity.size else 0


        # gazes
        robot_direction = np.array([odom["position"][2] for odom in episode["odom"]])
        peds_direction = np.array([[ped.theta for ped in peds] for peds in episode["peds"]])
        angle_robot_peds = np.squeeze(np.angle(np.array(peds_position - robot_position[:,np.newaxis]).view(np.complex128)))

        # time looking at pedestrians
        robot_gaze = Math.angle_difference(robot_direction[:,np.newaxis], angle_robot_peds)
        looking_at_frames = np.abs(robot_gaze) <= Config.ROBOT_GAZE_ANGLE
        total_time_looking_at_pedestrians = time[looking_at_frames.max(axis=1)].sum()
        time_looking_at_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looking_at_frames.T]
        
        # time being looked at by pedestrians
        ped_gaze = Math.angle_difference(peds_direction, np.pi - angle_robot_peds)
        looked_at_frames = np.abs(ped_gaze) <= Config.PEDESTRIAN_GAZE_ANGLE
        total_time_looked_at_by_pedestrians = time[looked_at_frames.max(axis=1)].sum()
        time_looked_at_by_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looked_at_frames.T]

        return PedsimMetric(
            **super_analysis,
            avg_velocity_in_personal_space = avg_velocity_in_personal_space,
            total_time_in_personal_space = total_time_in_personal_space,
            time_in_personal_space = time_in_personal_space,
            total_time_looking_at_pedestrians = total_time_looking_at_pedestrians,
            time_looking_at_pedestrians = time_looking_at_pedestrians,
            total_time_looked_at_by_pedestrians = total_time_looked_at_by_pedestrians,
            time_looked_at_by_pedestrians = time_looked_at_by_pedestrians,
            num_pedestrians = peds_position.shape[1]
        )