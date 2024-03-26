/*
 *  Copyright 2018, Magazino GmbH, Sebastian Pütz, Jorge Santos Simón
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  inter_action.cpp
 *
 *  authors:
 *    Sebastian Pütz <spuetz@uni-osnabrueck.de>
 *    Jorge Santos Simón <santos@magazino.eu>
 *
 */

#include <sstream>

#include "mbf_abstract_nav/inter_action.h"

namespace mbf_abstract_nav
{

InterAction::InterAction(
    const std::string &name,
    const mbf_utility::RobotInformation &robot_info)
  : AbstractActionBase(name, robot_info), path_seq_count_(0)
{
  ros::NodeHandle private_nh("~");
  // informative topics: current navigation goal
  current_goal_pub_ = private_nh.advertise<geometry_msgs::PoseStamped>("current_subgoal", 1);
}

void InterAction::runImpl(GoalHandle &goal_handle, AbstractInterExecution &execution)
{
  const mbf_msgs::GetInterPathGoal& goal = *(goal_handle.getGoal().get());

  mbf_msgs::GetInterPathResult result;
  geometry_msgs::PoseStamped start_pose;

  result.path.header.seq = path_seq_count_++;
  result.path.header.frame_id = robot_info_.getGlobalFrame();

  bool use_start_pose = goal.use_start_pose;
  current_goal_pub_.publish(goal.target_pose);

  bool inter_active = true;

  if(use_start_pose)
  {
    start_pose = goal.start_pose;
    const geometry_msgs::Point& p = start_pose.pose.position;
    ROS_DEBUG_STREAM_NAMED(name_, "Use the given start pose (" << p.x << ", " << p.y << ", " << p.z << ").");
  }
  else
  {
    // get the current robot pose
    if (!robot_info_.getRobotPose(start_pose))
    {
      result.outcome = mbf_msgs::GetInterPathResult::TF_ERROR;
      result.message = "Could not get the current robot pose!";
      goal_handle.setAborted(result, result.message);
      ROS_ERROR_STREAM_NAMED(name_, result.message << " Canceling the action call.");
      return;
    }
    else
    {
      const geometry_msgs::Point& p = start_pose.pose.position;
      ROS_DEBUG_STREAM_NAMED(name_, "Got the current robot pose at ("
          << p.x << ", " << p.y << ", " << p.z << ").");
    }
  }

  AbstractInterExecution::PlanningState state_inter_planning_input;

  std::vector<geometry_msgs::PoseStamped> plan, inter_plan;

  while (inter_active && ros::ok())
  {
    
    // get the current state of the inter_planning thread
    state_inter_planning_input = execution.getState();


    switch (state_inter_planning_input)
    {
      case AbstractInterExecution::INITIALIZED:
        ROS_DEBUG_STREAM_NAMED(name_, "inter state: initialized");
        execution.setNewPlan(goal.path.poses);
        if (!execution.start(start_pose, goal.target_pose))
        {
          result.outcome = mbf_msgs::GetInterPathResult::INTERNAL_ERROR;
          result.message = "Another thread is still inter_planning!";
          goal_handle.setAborted(result, result.message);
          ROS_ERROR_STREAM_NAMED(name_, result.message << " Canceling the action call.");
          inter_active = false;
        }
        break;

      case AbstractInterExecution::STARTED:
        ROS_DEBUG_STREAM_NAMED(name_, "inter state: started");
        break;

      case AbstractInterExecution::STOPPED:
        ROS_DEBUG_STREAM_NAMED(name_, "inter state: stopped");
        ROS_WARN_STREAM_NAMED(name_, "Planning has been stopped rigorously!");
        result.outcome = mbf_msgs::GetInterPathResult::STOPPED;
        result.message = "Inter has been stopped!";
        goal_handle.setAborted(result, result.message);
        inter_active = false;
        break;

      case AbstractInterExecution::CANCELED:
        ROS_DEBUG_STREAM_NAMED(name_, "inter state: canceled");
        ROS_DEBUG_STREAM_NAMED(name_, "Inter has been canceled successfully");
        result.path.header.stamp = ros::Time::now();
        result.outcome = mbf_msgs::GetInterPathResult::CANCELED;
        result.message = "Inter has been canceled!";
        goal_handle.setCanceled(result, result.message);
        inter_active = false;
        break;

        // in progress
      case AbstractInterExecution::PLANNING:
        if (execution.isPatienceExceeded())
        {
          ROS_INFO_STREAM_NAMED(name_, "Inter patience has been exceeded! Cancel inter_planning...");
          execution.cancel();
        }
        else
        {
          ROS_DEBUG_THROTTLE_NAMED(2.0, name_, "inter state: inter_planning");
        }
        break;

        // found a new plan
      case AbstractInterExecution::FOUND_PLAN:
        // set time stamp to now
        result.path.header.stamp = ros::Time::now();
        plan = execution.getPlan();

        ROS_DEBUG_STREAM_NAMED(name_, "inter state: found plan with cost: " << execution.getCost());

        if (!transformPlanToGlobalFrame(plan, inter_plan))
        {
          result.outcome = mbf_msgs::GetInterPathResult::TF_ERROR;
          result.message = "Could not transform the plan to the global frame!";

          ROS_ERROR_STREAM_NAMED(name_, result.message << " Canceling the action call.");
          goal_handle.setAborted(result, result.message);
          inter_active = false;
          break;
        }

        if (inter_plan.empty())
        {
          result.outcome = mbf_msgs::GetInterPathResult::EMPTY_PATH;
          result.message = "Inter returned an empty path!";

          ROS_ERROR_STREAM_NAMED(name_, result.message);
          goal_handle.setAborted(result, result.message);
          inter_active = false;
          break;
        }

        result.path.poses = inter_plan;
        result.cost = execution.getCost();
        result.outcome = execution.getOutcome();
        result.message = execution.getMessage();
        goal_handle.setSucceeded(result, result.message);

        inter_active = false;
        break;

        // no plan found
      case AbstractInterExecution::NO_PLAN_FOUND:
        ROS_DEBUG_STREAM_NAMED(name_, "inter state: no plan found");
        result.outcome = execution.getOutcome();
        result.message = execution.getMessage();
        goal_handle.setAborted(result, result.message);
        inter_active = false;
        break;

      case AbstractInterExecution::MAX_RETRIES:
        ROS_DEBUG_STREAM_NAMED(name_, "Inter reached the maximum number of retries");
        result.outcome = execution.getOutcome();
        result.message = execution.getMessage();
        goal_handle.setAborted(result, result.message);
        inter_active = false;
        break;

      case AbstractInterExecution::PAT_EXCEEDED:
        ROS_DEBUG_STREAM_NAMED(name_, "Inter exceeded the patience time");
        result.outcome = mbf_msgs::GetInterPathResult::PAT_EXCEEDED;
        result.message = "Inter exceeded the patience time";
        goal_handle.setAborted(result, result.message);
        inter_active = false;
        break;

      case AbstractInterExecution::INTERNAL_ERROR:
        ROS_FATAL_STREAM_NAMED(name_, "Internal error: Unknown error thrown by the plugin!"); // TODO getMessage from inter_planning
        inter_active = false;
        result.outcome = mbf_msgs::GetInterPathResult::INTERNAL_ERROR;
        result.message = "Internal error: Unknown error thrown by the plugin!";
        goal_handle.setAborted(result, result.message);
        break;

      default:
        result.outcome = mbf_msgs::GetInterPathResult::INTERNAL_ERROR;
        std::ostringstream ss;
        ss << "Internal error: Unknown state in a move base flex inter execution with the number: "
           << static_cast<int>(state_inter_planning_input);
        result.message = ss.str();
        ROS_FATAL_STREAM_NAMED(name_, result.message);
        goal_handle.setAborted(result, result.message);
        inter_active = false;
    }


    if (inter_active)
    {
      // try to sleep a bit
      // normally this thread should be woken up from the inter execution thread
      // in order to transfer the results to the controller.
      execution.waitForStateUpdate(boost::chrono::milliseconds(500));
    }
  }  // while (inter_active && ros::ok())

  if (!inter_active)
  {
    ROS_DEBUG_STREAM_NAMED(name_, "\"" << name_ << "\" action ended properly.");
  }
  else
  {
    ROS_ERROR_STREAM_NAMED(name_, "\"" << name_ << "\" action has been stopped!");
  }
}

bool InterAction::transformPlanToGlobalFrame(const std::vector<geometry_msgs::PoseStamped>& plan,
                                               std::vector<geometry_msgs::PoseStamped>& inter_plan)
{
  inter_plan.clear();
  inter_plan.reserve(plan.size());
  std::vector<geometry_msgs::PoseStamped>::const_iterator iter;
  bool tf_success = false;
  for (iter = plan.begin(); iter != plan.end(); ++iter)
  {
    geometry_msgs::PoseStamped global_pose;
    tf_success = mbf_utility::transformPose(robot_info_.getTransformListener(), robot_info_.getGlobalFrame(),
                                            robot_info_.getTfTimeout(), *iter, global_pose);
    if (!tf_success)
    {
      ROS_ERROR_STREAM("Can not transform pose from the \"" << iter->header.frame_id << "\" frame into the \""
                                                            << robot_info_.getGlobalFrame() << "\" frame !");
      return false;
    }
    inter_plan.push_back(global_pose);
  }
  return true;
}

} /* namespace mbf_abstract_nav */
