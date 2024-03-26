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
 *  abstract_inter_execution.cpp
 *
 *  authors:
 *    Sebastian Pütz <spuetz@uni-osnabrueck.de>
 *    Jorge Santos Simón <santos@magazino.eu>
 *
 */

#include "mbf_abstract_nav/abstract_inter_execution.h"

namespace mbf_abstract_nav
{

AbstractInterExecution::AbstractInterExecution(const std::string& name,
                                                   const mbf_abstract_core::AbstractInter::Ptr& inter_ptr,
                                                   const mbf_utility::RobotInformation &robot_info,
                                                   const ros::Publisher &goal_pub,
                                                   const MoveBaseFlexConfig& config)
  : AbstractExecutionBase(name, robot_info)
  , inter_(inter_ptr)
  , state_(INITIALIZED)
  , max_retries_(0)
  , interpolating_(false)
  , has_new_start_(false)
  , has_new_goal_(false)
  , global_goal_pub_(goal_pub)
{
  ros::NodeHandle private_nh("~");

  // non-dynamically reconfigurable parameters
  private_nh.param("robot_frame", robot_frame_, std::string("base_footprint"));
  private_nh.param("map_frame", global_frame_, std::string("map"));

  // dynamically reconfigurable parameters
  reconfigure(config);
}

AbstractInterExecution::~AbstractInterExecution()
{
}

template <typename _Iter>
double sumDistance(_Iter _begin, _Iter _end)
{
  // helper function to get the distance of a path.
  // in C++11, we could add static_assert on the interator_type.
  double dist = 0.;

  // minimum length of the path is 2.
  if (std::distance(_begin, _end) < 2)
    return dist;

  // two pointer iteration
  for (_Iter next = _begin + 1; next != _end; ++_begin, ++next)
    dist += mbf_utility::distance(*_begin, *next);

  return dist;
}

double AbstractInterExecution::getCost() const
{
  return cost_;
}

void AbstractInterExecution::reconfigure(const MoveBaseFlexConfig &config)
{
  boost::lock_guard<boost::mutex> guard(configuration_mutex_);

  max_retries_ = config.inter_max_retries;
  frequency_ = config.inter_frequency;

  // Timeout granted to the inter. We keep calling it up to this time or up to max_retries times
  // If it doesn't return within time, the navigator will cancel it and abort the corresponding action
  try
  {
    patience_ = ros::Duration(config.inter_patience);
  }
  catch (std::exception& ex)
  {
    ROS_ERROR_STREAM("Failed to set inter_patience: " << ex.what());
    patience_ = ros::Duration(0);
  }
}


typename AbstractInterExecution::PlanningState AbstractInterExecution::getState() const
{
  boost::lock_guard<boost::mutex> guard(state_mtx_);
  return state_;
}

void AbstractInterExecution::setState(PlanningState state, bool signalling)
{
  boost::lock_guard<boost::mutex> guard(state_mtx_);
  state_ = state;

  // we exit planning if we are signalling.
  interpolating_ = !signalling;

  // some states are quiet, most aren't
  if(signalling)
    condition_.notify_all();
}


ros::Time AbstractInterExecution::getLastValidPlanTime() const
{
  boost::lock_guard<boost::mutex> guard(plan_mtx_);
  return last_valid_plan_time_;
}


bool AbstractInterExecution::isPatienceExceeded() const
{
  return !patience_.isZero() && (ros::Time::now() - last_call_start_time_ > patience_);
}


std::vector<geometry_msgs::PoseStamped> AbstractInterExecution::getPlan() const
{
  boost::lock_guard<boost::mutex> guard(plan_mtx_);
  // copy plan and costs to output
  return inter_plan_;
}


void AbstractInterExecution::setNewGoal(const geometry_msgs::PoseStamped &goal)
{
  boost::lock_guard<boost::mutex> guard(goal_start_mtx_);
  goal_ = goal;
  has_new_goal_ = true;
}


void AbstractInterExecution::setNewStart(const geometry_msgs::PoseStamped &start)
{
  boost::lock_guard<boost::mutex> guard(goal_start_mtx_);
  start_ = start;
  has_new_start_ = true;
}


void AbstractInterExecution::setNewStartAndGoal(const geometry_msgs::PoseStamped &start,
                                                  const geometry_msgs::PoseStamped &goal)
{
  boost::lock_guard<boost::mutex> guard(goal_start_mtx_);
  start_ = start;
  goal_ = goal;
  has_new_start_ = true;
  has_new_goal_ = true;
}


bool AbstractInterExecution::start(const geometry_msgs::PoseStamped &start,
                                     const geometry_msgs::PoseStamped &goal)
{
  if (interpolating_)
  {
    return false;
  }
  boost::lock_guard<boost::mutex> guard(interpolating_mtx_);
  interpolating_ = true;
  start_ = start;
  goal_ = goal;

  const geometry_msgs::Point& s = start.pose.position;
  const geometry_msgs::Point& g = goal.pose.position;

  ROS_DEBUG_STREAM("Start planning from the start pose: (" << s.x << ", " << s.y << ", " << s.z << ")"
                                 << " to the goal pose: ("<< g.x << ", " << g.y << ", " << g.z << ")");

  return AbstractExecutionBase::start();
}


bool AbstractInterExecution::cancel()
{
  cancel_ = true; // force cancel immediately, as the call to cancel in the inter can take a while

  // returns false if cancel is not implemented or rejected by the inter (will run until completion)
  if (!inter_->cancel())
  {
    ROS_INFO_STREAM("Cancel planning failed or is not supported by the plugin. "
        << "Wait until the current planning finished!");

    return false;
  }
  return true;
}

uint32_t AbstractInterExecution::makePlan(const geometry_msgs::PoseStamped &start,
                                            const geometry_msgs::PoseStamped &goal,
                                            std::vector<geometry_msgs::PoseStamped> &plan,
                                            double &cost,
                                            std::string &message)
{
  return inter_->makePlan(start, goal, plan, cost, message);
}

void AbstractInterExecution::run()
{
  setState(STARTED, false);
  boost::lock_guard<boost::mutex> guard(interpolating_mtx_);
  int retries = 0;
  geometry_msgs::PoseStamped current_start = start_;
  geometry_msgs::PoseStamped current_goal = goal_;

  // init plan
  std::vector<geometry_msgs::PoseStamped> plan;
  if (!hasNewPlan())
  {
    setState(NO_GLOBAL_PLAN);
    interpolating_ = false;
    ROS_ERROR("robot navigation moving has no plan!");
  }

  last_call_start_time_ = ros::Time::now();
  last_valid_plan_time_ = ros::Time::now();

  try
  {
    while (interpolating_ && ros::ok())
    {
      // call the inter
      std::vector<geometry_msgs::PoseStamped> global_plan;
      double cost = 0.0;

      // lock goal start mutex
      goal_start_mtx_.lock();
      if (has_new_start_)
      {
        has_new_start_ = false;
        current_start = start_;
        ROS_INFO_STREAM("A new start pose is available. Planning with the new start pose!");
        const geometry_msgs::Point& s = start_.pose.position;
        ROS_INFO_STREAM("New planning start pose: (" << s.x << ", " << s.y << ", " << s.z << ")");
      }
      if (has_new_goal_)
      {
        has_new_goal_ = false;
        current_goal = goal_;
        ROS_INFO_STREAM("A new goal pose is available. Planning with the new goal pose.");
        const geometry_msgs::Point& g = goal_.pose.position;
        ROS_INFO_STREAM("New goal pose: (" << g.x << ", " << g.y << ", " << g.z << ")");
      }

      // unlock goal
      goal_start_mtx_.unlock();

      // update plan dynamically
      if (hasNewPlan())
      {
        global_plan = getNewPlan();
        // check if plan is empty
        if (global_plan.empty())
        {
          setState(EMPTY_GLOBAL_PLAN);
          interpolating_ = false;
          condition_.notify_all();
          return;
        }


        // check if plan could be set
        if (!inter_->setPlan(global_plan))
        {
          setState(INVALID_GLOBAL_PLAN);
          interpolating_ = false;
          condition_.notify_all();
          return;
        }

        global_goal_pub_.publish(global_plan.back());
      }

      if (cancel_)
      {
        ROS_INFO_STREAM("The inter has been canceled!");
        setState(CANCELED, true);
      }
      else
      {
        setState(PLANNING, false);

        outcome_ = makePlan(current_start, current_goal, plan, cost, message_);
        bool success = outcome_ < 10;

        boost::lock_guard<boost::mutex> guard(configuration_mutex_);

        if (cancel_ && !isPatienceExceeded())
        {
          ROS_INFO_STREAM("The inter \"" << name_ << "\" has been canceled!"); // but not due to patience exceeded
          setState(CANCELED, true);
        }
        else if (success)
        {
          ROS_DEBUG_STREAM("Successfully found a plan.");

          boost::lock_guard<boost::mutex> plan_mtx_guard(plan_mtx_);
          inter_plan_ = plan;
          cost_ = cost;
          // estimate the cost based on the distance if its zero.
          if (cost_ == 0)
            cost_ = sumDistance(inter_plan_.begin(), inter_plan_.end());

          last_valid_plan_time_ = ros::Time::now();
          setState(FOUND_PLAN, true);
        }
        else if (max_retries_ > 0 && ++retries > max_retries_)
        {
          ROS_INFO_STREAM("Planning reached max retries! (" << max_retries_ << ")");
          setState(MAX_RETRIES, true);
        }
        else if (isPatienceExceeded())
        {
          // Patience exceeded is handled at two levels: here to stop retrying planning when max_retries is
          // disabled, and on the navigation server when the inter doesn't return for more that patience seconds.
          // In the second case, the navigation server has tried to cancel planning (possibly without success, as
          // old nav_core-based inters do not support canceling), and we add here the fact to the log for info
          ROS_INFO_STREAM("Planning patience (" << patience_.toSec() << "s) has been exceeded"
                                                << (cancel_ ? "; inter canceled!" : ""));
          setState(PAT_EXCEEDED, true);
        }
        else if (max_retries_ == 0)
        {
          ROS_INFO_STREAM("Planning could not find a plan!");
          setState(NO_PLAN_FOUND, true);
        }
        else
        {
          ROS_DEBUG_STREAM("Planning could not find a plan! Trying again...");
        }
      }
    } // while (interpolating_ && ros::ok())
  }
  catch (const boost::thread_interrupted &ex)
  {
    // Inter thread interrupted; probably we have exceeded inter patience
    ROS_WARN_STREAM("Inter thread interrupted!");
    setState(STOPPED, true);
  }
  catch (...)
  {
    ROS_ERROR_STREAM("Unknown error occurred: " << boost::current_exception_diagnostic_information());
    setState(INTERNAL_ERROR, true);
  }
}

std::vector<geometry_msgs::PoseStamped> AbstractInterExecution::getNewPlan()
{
  boost::lock_guard<boost::mutex> guard(global_plan_mtx_);
  new_plan_ = false;
  return global_plan_;
}

void AbstractInterExecution::setNewPlan(
  const std::vector<geometry_msgs::PoseStamped> &plan)
{
  if (interpolating_)
  {
    // This is fine on continuous replanning
    ROS_DEBUG("Setting new plan while interpolating");
  }
  boost::lock_guard<boost::mutex> guard(global_plan_mtx_);
  new_plan_ = true;

  global_plan_ = plan;
}

bool AbstractInterExecution::hasNewPlan()
{
  boost::lock_guard<boost::mutex> guard(global_plan_mtx_);
  return new_plan_;
}

} /* namespace mbf_abstract_nav */

