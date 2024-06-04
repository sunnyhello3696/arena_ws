import gymnasium as gym
import torch as th
import rospy
from rosnav.utils.observation_space.observation_space_manager import (
    ObservationSpaceManager,
)
from rosnav.utils.observation_space.space_index import SPACE_INDEX
from torch import nn

from .base_extractor import RosnavBaseExtractor


from stable_baselines3.common.policies import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from typing import Dict
import math
import matplotlib.pyplot as plt
import os
import time



class ConvexExtractor_1d(RosnavBaseExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        if stacked_obs :
            print("ConvexExtractor_1d init with stacked_obs")
            rospy.loginfo("ConvexExtractor_1d init with stacked_obs")
        else:
            print("ConvexExtractor_1d init without stacked_obs")
            rospy.loginfo("ConvexExtractor_1d init without stacked_obs")
        
        self._goal_size, self._last_action_size,self._convex_map_size = (
            # observation_space_manager[SPACE_INDEX.LASER].shape[0],
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX].shape[0],
        )

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1
        super(ConvexExtractor_1d,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(self._num_stacks, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        # Compute shape by doing one forward pass
        with th.no_grad():
            desired_shape = (1, self._num_stacks, self._convex_map_size)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        # self.fc = nn.Sequential(
        #     nn.Linear(
        #         n_flatten
        #         + (self._goal_size + self._last_action_size) * self._num_stacks,
        #         self._features_dim,
        #     ),
        #     nn.ReLU(),
        # )
        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        _robot_state_size = self._goal_size + self._last_action_size
        if not self._stacked_obs:
            # observations in shape [batch_size, obs_size]
            convex_map = th.unsqueeze(observations[:, :-_robot_state_size], 1)
            robot_state = observations[:, -_robot_state_size:]

            cnn_features = self.cnn(convex_map)
            extracted_features = th.cat((cnn_features, robot_state), 1)
            return self.fc(extracted_features)
        else:
            # observations in shape [batch_size, num_stacks, obs_size]
            convex_map = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(1, 2)

            cnn_features = self.cnn(convex_map)
            extracted_features = th.cat((cnn_features, robot_state), 1)
            return self.fc(extracted_features)
        

class ConvexExtractor_2d(RosnavBaseExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        if stacked_obs :
            print("ConvexExtractor_2d init with stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init with stacked_obs")
        else:
            print("ConvexExtractor_2d init without stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init without stacked_obs")
        
        self._goal_size, self._last_action_size,self._convex_map_size = (
            # observation_space_manager[SPACE_INDEX.LASER].shape[0],
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX].shape[0],
        )

        # print("_goal_size: ",self._goal_size)
        # print("_last_action_size: ",self._last_action_size)
        # print("_convex_map_size: ",self._convex_map_size)

        self.convex_map_side = int(math.sqrt(self._convex_map_size))

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1
        super(ConvexExtractor_2d,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(self._num_stacks, 32, 5, 2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        _robot_state_size = self._goal_size + self._last_action_size
        
        if not self._stacked_obs:
            # For non-stacked observations
            convex_map_flat = observations[:, :-_robot_state_size]
            robot_state = observations[:, -_robot_state_size:]

            # Reshape convex_map_flat to [batch_size, num_channels, height, width]
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            
            # # 保存第一个批次的convex_map图像
            # self.save_convex_map(convex_map_reshaped, batch_idx=0)
            # self.print_partial_data(robot_state[0], name="Robot State", num_elements=5)

            
            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)
        else:
            # For stacked observations
            convex_map_flat = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(start_dim=1, end_dim=2)

            # Assuming each stack is concatenated along the last dimension, we need to split them and stack along a new dimension
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)

            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)

        return self.fc(extracted_features)

    
class ConvexExtractor_2d_cgd(RosnavBaseExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        if stacked_obs :
            print("ConvexExtractor_2d init with stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init with stacked_obs")
        else:
            print("ConvexExtractor_2d init without stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init without stacked_obs")
        
        self._goal_size, self._last_action_size,self._convex_map_size = (
            # observation_space_manager[SPACE_INDEX.LASER].shape[0],
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX].shape[0],
        )

        # print("_goal_size: ",self._goal_size)
        # print("_last_action_size: ",self._last_action_size)
        # print("_convex_map_size: ",self._convex_map_size)

        self.convex_map_side = int(math.sqrt(self._convex_map_size))

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1
        super(ConvexExtractor_2d_cgd,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        _robot_state_size = self._goal_size + self._last_action_size
        
        if not self._stacked_obs:
            # For non-stacked observations
            convex_map_flat = observations[:, :-_robot_state_size]
            robot_state = observations[:, -_robot_state_size:]

            # Reshape convex_map_flat to [batch_size, num_channels, height, width]
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            
            # # 保存第一个批次的convex_map图像
            # self.save_convex_map(convex_map_reshaped, batch_idx=0)
            # self.print_partial_data(robot_state[0], name="Robot State", num_elements=5)

            
            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)
        else:
            # For stacked observations
            convex_map_flat = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(start_dim=1, end_dim=2)

            # Assuming each stack is concatenated along the last dimension, we need to split them and stack along a new dimension
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)

            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)

        return self.fc(extracted_features)
    

class ConvexExtractor_2d_with_ActPts(RosnavBaseExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        if stacked_obs :
            print("ConvexExtractor_2d init with stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init with stacked_obs")
        else:
            print("ConvexExtractor_2d init without stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init without stacked_obs")
        
        self._goal_size, self._last_action_size,self._convex_map_size,self._last_action_points_size = (
            # observation_space_manager[SPACE_INDEX.LASER].shape[0],
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION_POINTS].shape[0],
        )

        print("_goal_size: ",self._goal_size)
        print("_last_action_size: ",self._last_action_size)
        print("_convex_map_size: ",self._convex_map_size)
        print("_last_action_points_size: ",self._last_action_points_size)

        self.convex_map_side = int(math.sqrt(self._convex_map_size))

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1
        super(ConvexExtractor_2d_with_ActPts,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(self._num_stacks, 32, 5, 2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size + self._last_action_points_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        _robot_state_size = self._goal_size + self._last_action_size + self._last_action_points_size
        
        if not self._stacked_obs:
            # For non-stacked observations
            convex_map_flat = observations[:, :-_robot_state_size]
            robot_state = observations[:, -_robot_state_size:]

            # Reshape convex_map_flat to [batch_size, num_channels, height, width]
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            
            # # 保存第一个批次的convex_map图像
            # self.save_convex_map(convex_map_reshaped, batch_idx=0)
            # self.print_partial_data(robot_state[0], name="Robot State", num_elements=5)

            
            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)
        else:
            # For stacked observations
            convex_map_flat = observations[:, :, :-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(start_dim=1, end_dim=2)

            # Assuming each stack is concatenated along the last dimension, we need to split them and stack along a new dimension
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)

            cnn_features = self.cnn(convex_map_reshaped)
            extracted_features = th.cat((cnn_features, robot_state), dim=1)

        return self.fc(extracted_features)
    
    # 假设这个方法在您的类中
    def save_convex_map(self, convex_map, batch_idx=0, save_path='/home/dmz/Pictures/convex_map'):
        """
        保存指定批次索引的convex_map图像到本地。

        参数:
        - convex_map: 卷积地图的张量。
        - batch_idx: 批次中要保存的样本的索引。
        - save_path: 保存图像的路径。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if convex_map.dim() == 4:  # [batch_size, num_channels, height, width]
            image = convex_map[batch_idx, 0].cpu().detach().numpy()  # 选取第一个通道的图像数据
            plt.imshow(image, cmap='gray')
            plt.axis('off')  # 不显示坐标轴

            # 构造保存图像的完整路径
            image_path = os.path.join(save_path, f'convex_map_{batch_idx}.png')
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            time.sleep(2)
            plt.close()  # 关闭图像，避免在notebook中显示
            print(f"Convex map saved to {image_path}")
        else:
            print("Unexpected convex_map dimension:", convex_map.dim())

    def print_partial_data(self, data, name="Data", num_elements=5):
        """
        打印指定数据的前几个元素。
        """
        print(f"{name} (partial):", data[:num_elements])


class ConvexExtractor_2d_with_ActPts_with_ConvexQueue(RosnavBaseExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        if stacked_obs :
            print("ConvexExtractor_2d init with stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init with stacked_obs")
        else:
            print("ConvexExtractor_2d init without stacked_obs")
            rospy.loginfo("ConvexExtractor_2d init without stacked_obs")
        
        self._goal_size,self._convex_map_size,self._last_action_points_size,self._convex_queue_size = (
            observation_space_manager[SPACE_INDEX.GOAL].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX].shape[0],
            observation_space_manager[SPACE_INDEX.LAST_ACTION_POINTS].shape[0],
            observation_space_manager[SPACE_INDEX.CONVEX_QUEUE].shape[0],
        )

        print("_goal_size: ",self._goal_size)
        print("_convex_map_size: ",self._convex_map_size)
        print("_last_action_points_size: ",self._last_action_points_size)
        print("_convex_queue_size: ",self._convex_queue_size)

        self.convex_map_side = int(math.sqrt(self._convex_map_size))

        self._num_stacks = observation_space.shape[0] if stacked_obs else 1
        super(ConvexExtractor_2d_with_ActPts_with_ConvexQueue,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn_1d = nn.Sequential(
            nn.Conv1d(self._num_stacks, 32, 5, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

            desired_shape_1d = (1, self._num_stacks, self._convex_queue_size)
            tensor_forward_1d = th.randn(desired_shape_1d)
            n_flatten_1d = self.cnn_1d(tensor_forward_1d).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                (n_flatten+n_flatten_1d)
                + (self._goal_size + self._last_action_points_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        # 注意顺序
        observation_spaces = [
            SPACE_INDEX.CONVEX,
            SPACE_INDEX.CONVEX_QUEUE,
            SPACE_INDEX.GOAL,
            SPACE_INDEX.LAST_ACTION_POINTS,
        ]
        """
        _robot_state_size = self._goal_size + self._last_action_points_size
        
        if not self._stacked_obs:
            # For non-stacked observations
            convex_map_flat = observations[:, :(-_robot_state_size-self._convex_queue_size)]
            convex_queue = th.unsqueeze(observations[:, (-_robot_state_size-self._convex_queue_size):-_robot_state_size], 1)
            robot_state = observations[:, -_robot_state_size:]

            # Reshape convex_map_flat to [batch_size, num_channels, height, width]
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            
            # # 保存第一个批次的convex_map图像
            # self.save_convex_map(convex_map_reshaped, batch_idx=0)
            # self.print_partial_data(robot_state[0], name="Robot State", num_elements=5)
            # self.print_partial_data(convex_queue[0],name="Convex Queue",num_elements=5)

            
            cnn_features = self.cnn(convex_map_reshaped)
            cnn_features_1d = self.cnn_1d(convex_queue)
            extracted_features = th.cat((cnn_features, cnn_features_1d, robot_state), dim=1)
        else:
            # For stacked observations
            convex_map_flat = observations[:, :, :(-_robot_state_size-self._convex_queue_size)]
            convex_queue = observations[:, :, (-_robot_state_size-self._convex_queue_size):-_robot_state_size]
            robot_state = observations[:, :, -_robot_state_size:].flatten(start_dim=1, end_dim=2)

            # Assuming each stack is concatenated along the last dimension, we need to split them and stack along a new dimension
            convex_map_reshaped = convex_map_flat.view(-1, self._num_stacks, self.convex_map_side, self.convex_map_side)

            cnn_features = self.cnn(convex_map_reshaped)
            cnn_features_1d = self.cnn_1d(convex_queue)
            extracted_features = th.cat((cnn_features, cnn_features_1d, robot_state), dim=1)

        return self.fc(extracted_features)
    
    # 假设这个方法在您的类中
    def save_convex_map(self, convex_map, batch_idx=0, save_path='/home/dmz/Pictures/convex_map'):
        """
        保存指定批次索引的convex_map图像到本地。

        参数:
        - convex_map: 卷积地图的张量。
        - batch_idx: 批次中要保存的样本的索引。
        - save_path: 保存图像的路径。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if convex_map.dim() == 4:  # [batch_size, num_channels, height, width]
            image = convex_map[batch_idx, 0].cpu().detach().numpy()  # 选取第一个通道的图像数据
            plt.imshow(image, cmap='gray')
            plt.axis('off')  # 不显示坐标轴

            # 构造保存图像的完整路径
            image_path = os.path.join(save_path, f'convex_map_{batch_idx}.png')
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
            time.sleep(2)
            plt.close()  # 关闭图像，避免在notebook中显示
            print(f"Convex map saved to {image_path}")
        else:
            print("Unexpected convex_map dimension:", convex_map.dim())

    def print_partial_data(self, data, name="Data", num_elements=5):
        """
        打印指定数据的前几个元素。
        """
        print(f"{name} (partial):", data[:num_elements])
    

class ConvexExtractor_2d_with_ActPts_2(ConvexExtractor_2d_with_ActPts):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        # init father class
        super(ConvexExtractor_2d_with_ActPts_2,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(self._num_stacks, 32, 5, 2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size + self._last_action_points_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )

class ConvexExtractor_2d_with_ActPts_3(ConvexExtractor_2d_with_ActPts):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        observation_space_manager: ObservationSpaceManager,
        features_dim: int = 256,
        stacked_obs: bool = False,
        *args,
        **kwargs
    ) -> None:
        
        # init father class
        super(ConvexExtractor_2d_with_ActPts_3,self).__init__(
            observation_space=observation_space,
            observation_space_manager=observation_space_manager,
            features_dim=features_dim,
            stacked_obs=stacked_obs,
        )

    def _setup_network(self):
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(self._num_stacks, 32, 5, 2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(self._num_stacks, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # desired_shape = (1, self._num_stacks, self._convex_map_size)
            desired_shape = (1, self._num_stacks, self.convex_map_side, self.convex_map_side)
            tensor_forward = th.randn(desired_shape)
            n_flatten = self.cnn(tensor_forward).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(
                n_flatten
                + (self._goal_size + self._last_action_size + self._last_action_points_size) * self._num_stacks,
                self._features_dim,
            ), 
            nn.ReLU(),
        )