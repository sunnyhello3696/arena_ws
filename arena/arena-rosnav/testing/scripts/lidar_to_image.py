import numpy as np
import math
from collections import deque
import cv2
import time

import matplotlib.pyplot as plt

class Lidar2Image():

    def __init__(
        self, *args, **kwargs
    ) -> None:
        self.map_size = 256
        self.lidar_max_range = 4.0
        self.xy_resolution = (self.lidar_max_range*2)/self.map_size

    def read_lidar_data(self, filepath):
        """
        Reads x and y coordinates from a file.
        """
        x, y = np.loadtxt(filepath, delimiter=',', unpack=True)
        return x, y

    def lidar_convex_to_map(self, lidar_filepath, convex_filepath) -> np.ndarray:
        """
        Convert the laser scan data to a 2D map.

        Args:
            laser_scan (np.ndarray): The laser scan data.
            laser_convex (Galaxy2D): The laser convex data.

        Returns:
            np.ndarray: The 2D map.
        """

        # read x_lidar, y_lidar from file
        x_lidar, y_lidar = self.read_lidar_data(lidar_filepath)
        # for i in range(len(x_lidar)):
        #     print(f"{x_lidar[i]}, {y_lidar[i]}")
        x_convex, y_convex = self.read_lidar_data(convex_filepath)

        x_convex = -x_convex  # 按x轴翻转
        y_convex = -y_convex  # 按y轴翻转

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
        
        start_time = time.time()
        lidar_map = self.occupancy_fill(x_lidar, y_lidar, occupancy_map, min_x, min_y,max_x,max_y, center_x, center_y, x_w, y_w,
                               fill_value=128, method='fillPoly')
        occupancy_map = np.full((x_w, y_w), 255, dtype=np.uint8)
        convex_map = self.occupancy_fill(x_convex, y_convex, occupancy_map, min_x, min_y,max_x,max_y, center_x, center_y,
                                x_w, y_w, fill_value=0, method='fillPoly')
        end_time = time.time() - start_time
        print(f"method took: {end_time:.5f} seconds")
        
        # Merge lidar_map and convex_map, giving priority to convex_map
        merged_map = np.where(convex_map == 0, 0, lidar_map) 
        return merged_map, lidar_map, convex_map
    
    
    def occupancy_fill(self, ox, oy, occupancy_map, min_x, min_y,max_x,max_y, center_x, center_y, x_w, y_w, fill_value=0, method='fillPoly'):
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
        elif method == 'fillPoly':
            # Convert real-world coordinates to image coordinates
            ox_img = ((ox - min_x) / self.xy_resolution).astype(int)
            oy_img = ((oy - min_y) / self.xy_resolution).astype(int)
            # Create a polygon from the convex hull points and fill it
            points = np.vstack([ox_img, oy_img]).T.reshape((-1, 1, 2))
            cv2.fillPoly(occupancy_map, [points], color=(fill_value))
            # cv2.fillConvexPoly(occupancy_map, points, color=(fill_value))
            
            # Rotate the image clockwise 90 degrees
            occupancy_map = cv2.rotate(occupancy_map, cv2.ROTATE_90_CLOCKWISE)
            # Flip the image along the y-axis
            occupancy_map = cv2.flip(occupancy_map, 1)
        return occupancy_map
    
    def init_flood_fill(self, center_point, obstacle_points, occupancy_map, fill_value, xy_points, min_coord):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / self.xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / self.xy_resolution))
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
    

    def visualize_maps(self, lidar_map, convex_map, merged_map):
        """
        Visualizes lidar, convex, and merged maps with x and y axis ranges set from -4 to +4,
        without altering the display of the images.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        cmap = 'gray'  # Define the color map to be used
        extent = [-4, 4, -4, 4]  # Set the extent of the axes [xmin, xmax, ymin, ymax]

        # Lidar Map
        lidar = axs[0].imshow(lidar_map, cmap=cmap, extent=extent)
        axs[0].set_title('Lidar Map')
        fig.colorbar(lidar, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        lidar.set_clim(0, 255)

        # Convex Map
        convex = axs[1].imshow(convex_map, cmap=cmap, extent=extent)
        axs[1].set_title('Convex Map')
        fig.colorbar(convex, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        convex.set_clim(0, 255)

        # Merged Map
        merged = axs[2].imshow(merged_map, cmap=cmap, extent=extent)
        axs[2].set_title('Merged Map')
        fig.colorbar(merged, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        merged.set_clim(0, 255)

        plt.show()


# main
# Usage example
lidar_to_image = Lidar2Image()
lidar_filepath = '/home/dmz/arena_ws/src/arena/arena-rosnav/testing/scripts/lidar_filepath.txt'
convex_filepath = '/home/dmz/arena_ws/src/arena/arena-rosnav/testing/scripts/convex_filepath.txt'

# You'll need to replace 'lidar_filepath' and 'convex_filepath' with the actual file paths
merged_map, lidar_map, convex_map = lidar_to_image.lidar_convex_to_map(lidar_filepath, convex_filepath)
lidar_to_image.visualize_maps(lidar_map, convex_map, merged_map)
