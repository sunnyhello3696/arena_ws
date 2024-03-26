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
 *  costmap_inter_execution.cpp
 *
 *  authors:
 *    Sebastian Pütz <spuetz@uni-osnabrueck.de>
 *    Jorge Santos Simón <santos@magazino.eu>
 *
 */
#include <mbf_msgs/GetInterPathResult.h>

#include "mbf_costmap_nav/costmap_inter_execution.h"

namespace mbf_costmap_nav
{
CostmapInterExecution::CostmapInterExecution(const std::string& inter_name,
                                                 const mbf_costmap_core::CostmapInter::Ptr& inter_ptr,
                                                 const mbf_utility::RobotInformation& robot_info,
                                                 const ros::Publisher &goal_pub,
                                                 const CostmapWrapper::Ptr& global_costmap_ptr,
                                                 const CostmapWrapper::Ptr& local_costmap_ptr,
                                                 const MoveBaseFlexConfig& config)
  : AbstractInterExecution(inter_name, inter_ptr, robot_info, goal_pub, toAbstract(config)), global_costmap_ptr_(global_costmap_ptr), local_costmap_ptr_(local_costmap_ptr)
{
  ros::NodeHandle private_nh("~");
  private_nh.param("inter_lock_costmap", lock_costmap_, true);
}

CostmapInterExecution::~CostmapInterExecution()
{
}

mbf_abstract_nav::MoveBaseFlexConfig CostmapInterExecution::toAbstract(const MoveBaseFlexConfig &config)
{
  // copy the inter-related abstract configuration common to all MBF-based navigation
  mbf_abstract_nav::MoveBaseFlexConfig abstract_config;
  abstract_config.inter_frequency = config.inter_frequency;
  abstract_config.inter_patience = config.inter_patience;
  abstract_config.inter_max_retries = config.inter_max_retries;
  return abstract_config;
}

uint32_t CostmapInterExecution::makePlan(const geometry_msgs::PoseStamped &start,
                                           const geometry_msgs::PoseStamped &goal,
                                           std::vector<geometry_msgs::PoseStamped> &plan,
                                           double &cost,
                                           std::string &message)
{
  // transform the input to the global frame of the costmap, since this is an
  // "implicit" requirement for most inters
  // note: costmap_2d::Costmap2DROS::getTransformTolerance might be a good idea,
  // but it's not part of the class API in ros-kinetic
  //return 0;

  const ros::Duration timeout(0.5);
  const std::string frame = global_costmap_ptr_->getGlobalFrameID();
  geometry_msgs::PoseStamped g_start, g_goal;

  if (!mbf_utility::transformPose(robot_info_.getTransformListener(), frame, timeout, start, g_start))
    return mbf_msgs::GetInterPathResult::TF_ERROR;

  if (!mbf_utility::transformPose(robot_info_.getTransformListener(), frame, timeout, goal, g_goal))
    return mbf_msgs::GetInterPathResult::TF_ERROR;

  if (lock_costmap_)
  {
    boost::unique_lock<costmap_2d::Costmap2D::mutex_t> global_lock(*(global_costmap_ptr_->getCostmap()->getMutex()));
    boost::unique_lock<costmap_2d::Costmap2D::mutex_t> local_lock(*(local_costmap_ptr_->getCostmap()->getMutex()));
    return inter_->makePlan(g_start, g_goal, plan, cost, message);
  }
  return inter_->makePlan(g_start, g_goal, plan, cost, message);
}

} /* namespace mbf_costmap_nav */
