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
# import ros_numpy


@SpaceFactory.register("convex")
class ConvexSpace(BaseObservationSpace):
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
        convex_map_size: int = 256,
        *args, 
        **kwargs,
    ) -> None:
        self.map_size = convex_map_size
        self.lidar_max_range = 9.0
        self.xy_resolution = (self.lidar_max_range*2)/self.map_size
        self.scale_factor = 1.0

        self._space = self.get_gym_space()
        self.enable_rviz = False

        # if debug mode
        if rospy.get_param("/debug_mode", False):
            print(f"debug_mode: {rospy.get_param('/debug_mode')}")
            self.rviz_lidar_pub = rospy.Publisher("/laser_convex_map/lidar", Image, queue_size=1)
            self.rviz_convex_pub = rospy.Publisher("/laser_convex_map/convex", Image, queue_size=1)
            self.rviz_merged_pub = rospy.Publisher("/laser_convex_map/merged", Image, queue_size=1)
            self.enable_rviz = rospy.get_param("/if_viz", False)
            print(f"enable_rviz: {self.enable_rviz}")

        print("ConvexSpace init")
        rospy.loginfo("ConvexSpace init")

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
            high=30,
            shape=(self.map_size * self.map_size,),
            dtype=np.float32,
        )

    def encode_observation(self, observation: dict, *args, **kwargs) -> ndarray:
        """
        Encodes the laser scan observation.

        Args:
            observation (dict): The observation dictionary.

        Returns:
            ndarray: The encoded laser scan observation.
        """
        # observation[OBS_DICT_KEYS.LASER_CONVEX]: type: np.array([])
        laser_convex = observation[OBS_DICT_KEYS.LASER_CONVEX]

        # observation[OBS_DICT_KEYS.LASER]: np.array(sensor_msgs.msg.LaserScan.ranges, np.float32)
        laser_scan = observation[OBS_DICT_KEYS.LASER]
        laser_convex_map = self.lidar_convex_to_map(laser_scan, laser_convex).flatten()

        # Convert to np.float32 and scale values from 0-255 to 0-30
        laser_convex_map = (laser_convex_map.astype(np.float32) / 255.0) * 30.0
        
        # Assert that the flattened map has the expected length
        expected_length = self.map_size * self.map_size
        assert laser_convex_map.size == expected_length, f"Expected laser_convex_map size to be {expected_length}, got {laser_convex_map.size}"

        return laser_convex_map


    def lidar_convex_to_map(self, laser_scan: np.ndarray, laser_convex: np.ndarray) -> np.ndarray:
        """
        Convert the laser scan data to a 2D map.

        Args:
            laser_scan (np.ndarray): The laser scan data.
            laser_convex (np.ndarray): The laser convex data.

        Returns:
            np.ndarray: The 2D map.
        """
        # convex_vertex == np.array([]) -> is_convex_reliable= false
        if len(laser_convex) == 0:
            is_convex_reliable = False
        else:
            is_convex_reliable = True

        x_lidar, y_lidar = self.lidar_ranges_to_xy(laser_scan)
        x_lidar = np.append(x_lidar, 0)
        y_lidar = np.append(y_lidar, 0)
        if is_convex_reliable:
            x_convex, y_convex = self.convex_vertex_to_xy(laser_convex)
            x_convex = np.append(x_convex, 0)
            y_convex = np.append(y_convex, 0)
            x_convex = -x_convex  # 按x轴翻转
            y_convex = -y_convex  # 按y轴翻转

        # print("=================================================================")
        # for x_val, y_val in zip(x_lidar, y_lidar):
        #     print(f"{x_val:.3f},{y_val:.3f}")
        # print("**********************")
        # if is_convex_reliable:
        #     for x_val, y_val in zip(x_convex, y_convex):
        #         print(f"{x_val:.3f},{y_val:.3f}")
        # print("=================================================================")

        min_x = -self.lidar_max_range
        max_x = self.lidar_max_range
        min_y = -self.lidar_max_range
        max_y = self.lidar_max_range
        x_w = self.map_size
        y_w = self.map_size
        center_x = int(
            round(-min_x / self.xy_resolution))  # center x coordinate of the grid map
        center_y = int(
            round(-min_y / self.xy_resolution))  # center y coordinate of the grid map
        
        occupancy_map = np.full((x_w, y_w), 255, dtype=np.uint8)
        lidar_map = self.occupancy_fill(x_lidar, y_lidar, occupancy_map, min_x, min_y, center_x, center_y, x_w, y_w,
                               fill_value=128, method='fillPoly')
        occupancy_map = np.full((x_w, y_w), 255, dtype=np.uint8)
        if is_convex_reliable:
            convex_map = self.occupancy_fill(x_convex, y_convex, occupancy_map, min_x, min_y, center_x, center_y,
                                x_w, y_w, fill_value=0, method='fillPoly')
            # 如果在convex_map中某个位置的值为0，则在merged_map的相同位置上设置为0；
            # 如果在convex_map中某个位置的值不为0，则在merged_map的相同位置上采用lidar_map中对应位置的值。
            merged_map = np.where(convex_map == 0, 0, lidar_map) 
            if self.enable_rviz:
                self.publish_to_rviz(lidar_map,convex_map,merged_map)

            return merged_map
        else:
            rospy.logwarn("Convex data is not reliable, returning lidar_map")
            return lidar_map
    

    def lidar_ranges_to_xy(self, laser_scan: np.ndarray) -> np.ndarray:
        """
        Convert the laser scan ranges to x, y coordinates.

        Args:
            laser_scan (np.ndarray): The laser scan data.

        Returns:
            np.ndarray: The x, y coordinates.
        """
        angles = np.linspace(-np.pi, np.pi, len(laser_scan))
        x = laser_scan * np.cos(angles)
        y = laser_scan * np.sin(angles)
        return x, y
    
    def convex_vertex_to_xy(self, laser_convex: np.ndarray) -> np.ndarray:
        """
        Convert the laser convex data to x, y coordinates.

        Args:
            laser_convex (np.ndarray): The laser convex data.
            laser_convex = np.array(
                [
                    [point.x, point.y]
                    for point in convex_vertex.points
                ]
            )

        Returns:
            np.ndarray: The x, y coordinates.
        """

        x = laser_convex[:, 0]
        y = laser_convex[:, 1]
        return np.array(x), np.array(y)
    
    def occupancy_fill(self, ox, oy, occupancy_map, min_x, min_y, center_x, center_y, x_w, y_w, fill_value=0, method='fillPoly'):
        # occupancy grid computed with bresenham ray casting
        if method == 'bresenham':
            for (x, y) in zip(ox, oy):
                # 将障碍物的真实世界坐标(x,y)转换为栅格地图上的坐标(ix,iy)
                # x coordinate of the occupied area
                ix = int(round((x - min_x) / self.xy_resolution))
                # y coordinate of the occupied area
                iy = int(round((y - min_y) / self.xy_resolution))

                laser_beams = self.bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point
                for laser_beam in laser_beams:
                    occupancy_map[laser_beam[0]][laser_beam[1]] = fill_value  # free area
                # occupancy_map[ix][iy] = 255  # occupied area 255
                # occupancy_map[ix + 1][iy] = 255  # extend the occupied area
                # occupancy_map[ix][iy + 1] = 255  # extend the occupied area
                # occupancy_map[ix + 1][iy + 1] = 255  # extend the occupied area

        # occupancy grid computed with flood fill
        elif method == 'floodfill':
            occupancy_map = self.init_flood_fill((center_x, center_y),
                                            (ox, oy),
                                            occupancy_map,
                                            fill_value,
                                            (x_w, y_w),
                                            (min_x, min_y))
            self.flood_fill((center_x, center_y), occupancy_map, fill_value=fill_value)
            occupancy_map = np.array(occupancy_map, dtype=np.uint8)
            for (x, y) in zip(ox, oy):
                ix = int(round((x - min_x) / self.xy_resolution))
                iy = int(round((y - min_y) / self.xy_resolution))
                # occupancy_map[ix][iy] = 255  # occupied area 255
                # occupancy_map[ix + 1][iy] = 255  # extend the occupied area
                # occupancy_map[ix][iy + 1] = 255  # extend the occupied area
                # occupancy_map[ix + 1][iy + 1] = 255  # extend the occupied area
        elif method == 'fillPoly':
            # Convert real-world coordinates to image coordinates
            ox_img = ((ox - min_x) / self.xy_resolution).astype(int)
            oy_img = ((oy - min_y) / self.xy_resolution).astype(int)
            # Create a polygon from the convex hull points and fill it
            points = np.vstack([ox_img, oy_img]).T.reshape((-1, 1, 2))
            cv2.fillPoly(occupancy_map, [points], color=(fill_value))
            # Rotate the image clockwise 90 degrees
            occupancy_map = cv2.rotate(occupancy_map, cv2.ROTATE_90_CLOCKWISE)
            # Flip the image along the y-axis
            occupancy_map = cv2.flip(occupancy_map, 1)
        
        return occupancy_map
    
    def init_flood_fill(self, center_point, obstacle_points, occupancy_map, fill_value, xy_points, min_coord):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        fill_value: value to fill the map with
        xy_points: (x_w, y_w)
        min_coord: minimum x and y coordinates
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        for (x, y) in zip(ox, oy):

            # 将真实世界坐标(x,y)转换为栅格地图上的坐标(ix,iy)
            ix = int(round((x - min_x) / self.xy_resolution))
            iy = int(round((y - min_y) / self.xy_resolution))

            # 用Bresenham算法计算从前一个点到当前点的连续区域
            free_area = self.bresenham((prev_ix, prev_iy), (ix, iy))

            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = fill_value
            prev_ix = ix
            prev_iy = iy
        return occupancy_map


    def flood_fill(self, center_point, occupancy_map, fill_value=0):
        """
        填充或标记占用栅格地图中的连续区域。该算法从一个指定的起始点开始，然后向四个方向（东、西、南、北）扩展，直到遇到边界或已被标记的区域。
        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()  # 创建一个双端队列 fringe，用于存储待处理的点。
        fringe.appendleft(center_point)  # 将起始点加入队列
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 255:
                    occupancy_map[nx - 1, ny] = fill_value
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 255:
                    occupancy_map[nx + 1, ny] = fill_value
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 255:
                    occupancy_map[nx, ny - 1] = fill_value
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 255:
                    occupancy_map[nx, ny + 1] = fill_value
                    fringe.appendleft((nx, ny + 1))

    def bresenham(self, start, end):
        """
        用来描绘由两点所决定的直线的算法，它会算出一条线段在n维位图上最接近的点。
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        # >>> points1 = bresenham((4, 4), (6, 10))
        # >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points
    
    def publish_to_rviz(self, laser_map: np.ndarray, convex_map: np.ndarray, merged_map: np.ndarray):
        # Ensure laser_convex_map is two-dimensional for mono8 encoding
        laser_map_squeezed = np.squeeze(laser_map).astype(np.uint8)
        convex_map_squeezed = np.squeeze(convex_map).astype(np.uint8)
        merged_map_squeezed = np.squeeze(merged_map).astype(np.uint8)

        # # If needed, perform resizing
        # if self.scale_factor != 1.0:
        #     height, width = laser_map_squeezed.shape[:2]
        #     new_dimensions = (int(width * self.scale_factor), int(height * self.scale_factor))
        #     laser_map_squeezed = cv2.resize(laser_map_squeezed, new_dimensions, interpolation=cv2.INTER_AREA)
        #     convex_map_squeezed = cv2.resize(convex_map_squeezed, new_dimensions, interpolation=cv2.INTER_AREA)

        # # Convert numpy array to ROS Image message
        # ros_image = self.bridge.cv2_to_imgmsg(laser_map_squeezed, encoding="mono8")
        # self.rviz_lidar_pub.publish(ros_image)

        # ros_image = self.bridge.cv2_to_imgmsg(convex_map_squeezed, encoding="mono8")
        # self.rviz_convex_pub.publish(ros_image)

        # # Prepare the ROS Image messages
        # laser_ros_image = ros_numpy.msgify(Image, laser_map_squeezed, encoding='mono8')
        # convex_ros_image = ros_numpy.msgify(Image, convex_map_squeezed, encoding='mono8')

        # # Publish the ROS Image messages
        # self.rviz_lidar_pub.publish(laser_ros_image)
        # self.rviz_convex_pub.publish(convex_ros_image)

        # Manually construct the ROS Image message for laser_map
        laser_ros_image = Image()
        laser_ros_image.header.stamp = rospy.Time.now()
        laser_ros_image.height = laser_map_squeezed.shape[0]
        laser_ros_image.width = laser_map_squeezed.shape[1]
        laser_ros_image.encoding = "mono8"
        laser_ros_image.is_bigendian = 0
        laser_ros_image.step = laser_ros_image.width
        laser_ros_image.data = laser_map_squeezed.tobytes()

        # Publish the laser_map image
        self.rviz_lidar_pub.publish(laser_ros_image)

        # Repeat the process for convex_map
        convex_ros_image = Image()
        convex_ros_image.header.stamp = rospy.Time.now()
        convex_ros_image.height = convex_map_squeezed.shape[0]
        convex_ros_image.width = convex_map_squeezed.shape[1]
        convex_ros_image.encoding = "mono8"
        convex_ros_image.is_bigendian = 0
        convex_ros_image.step = convex_ros_image.width
        convex_ros_image.data = convex_map_squeezed.tobytes()

        # Publish the convex_map image
        self.rviz_convex_pub.publish(convex_ros_image)

        # Repeat the process for merged_map
        merged_ros_image = Image()
        merged_ros_image.header.stamp = rospy.Time.now()
        merged_ros_image.height = merged_map_squeezed.shape[0]
        merged_ros_image.width = merged_map_squeezed.shape[1]
        merged_ros_image.encoding = "mono8"
        merged_ros_image.is_bigendian = 0
        merged_ros_image.step = merged_ros_image.width
        merged_ros_image.data = merged_map_squeezed.tobytes()
        
        # Publish the merged_map image
        self.rviz_merged_pub.publish(merged_ros_image)
