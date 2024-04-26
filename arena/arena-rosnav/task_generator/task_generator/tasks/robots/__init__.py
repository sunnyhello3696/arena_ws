import rospy
from task_generator.constants import Config
from task_generator.shared import PositionOrientation
from task_generator.tasks import TaskMode

from typing import Any, Dict

class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Args:
        **kwargs: Additional keyword arguments.

    Attributes:
        _PROPS (TaskProperties): Task properties object.

    """

    _last_reset: int
    _done_info: Dict[str, Any]

    def __init__(self, **kwargs):
        TaskMode.__init__(self, **kwargs)
        self._done_info = {}
        self._TIMEOUT_INFO = {
            "is_done": True,
            "done_reason": 0,
            "is_success": False,
        }

    def reset(self, **kwargs):
        self._last_reset = self._PROPS.clock.clock.secs
        self._done_info = {}

    def set_position(self, position: PositionOrientation):
        """
        Set the position of all robots.

        Args:
            position (PositionOrientation): The desired position and orientation.

        """
        for robot in self._PROPS.robot_managers:
            robot.reset(position, None)

    def set_goal(self, position: PositionOrientation):
        """
        Set the goal position for all robots.

        Args:
            position (PositionOrientation): The desired goal position and orientation.

        """
        for robot in self._PROPS.robot_managers:
            robot.reset(None, position)

    @property
    def done(self):
        """
        Check if all robots have completed their tasks.

        Returns:
            bool: True if all robots are done, False otherwise.

        """
        # convert to float and keep only 2 decimal places
        episode_duration = round(float(self._PROPS.clock.clock.secs - self._last_reset), 2)
        self._done_info.update({"episode_time": episode_duration})
        if (episode_duration) > 80.0:
            self._done_info.update(self._TIMEOUT_INFO)
            rospy.loginfo(f"Timeout for task")
            return True
        
        # return all(robot.is_done for robot in self._PROPS.robot_managers)
        # check self._PROPS.robot_managers length = 1
        if len(self._PROPS.robot_managers) == 1:
            if self._PROPS.robot_managers[0].is_done:
                self._done_info.update(self._PROPS.robot_managers[0].get_done_info())
                return True
        else:
            rospy.logerr("TM_Robots: done property is not implemented for multiple robots")
            raise NotImplementedError("TM_Robots: done property is not implemented for multiple robots")
    
    def get_done_info(self):
        """
        Get the done information for all robots.

        Returns:
            Dict[str, Any]: The done information for all robots.

        """
        return self._done_info
