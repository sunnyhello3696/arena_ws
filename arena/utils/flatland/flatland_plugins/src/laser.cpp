/*
 *  ______                   __  __              __
 * /\  _  \           __    /\ \/\ \            /\ \__
 * \ \ \L\ \  __  __ /\_\   \_\ \ \ \____    ___\ \ ,_\   ____
 *  \ \  __ \/\ \/\ \\/\ \  /'_` \ \ '__`\  / __`\ \ \/  /',__\
 *   \ \ \/\ \ \ \_/ |\ \ \/\ \L\ \ \ \L\ \/\ \L\ \ \ \_/\__, `\
 *    \ \_\ \_\ \___/  \ \_\ \___,_\ \_,__/\ \____/\ \__\/\____/
 *     \/_/\/_/\/__/    \/_/\/__,_ /\/___/  \/___/  \/__/\/___/
 * @copyright Copyright 2017 Avidbots Corp.
 * @name	  laser.cpp
 * @brief   Laser plugin
 * @author  Chunshang Li
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2017, Avidbots Corp.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Avidbots Corp. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <flatland_plugins/laser.h>
#include <flatland_server/collision_filter_registry.h>
#include <flatland_server/exceptions.h>
#include <flatland_server/model_plugin.h>
#include <flatland_server/yaml_reader.h>
#include <geometry_msgs/TransformStamped.h>
#include <pluginlib/class_list_macros.h>
#include <boost/algorithm/string/join.hpp>
#include <cmath>
#include <limits>

using namespace flatland_server;

namespace flatland_plugins {

void Laser::OnInitialize(const YAML::Node &config) {
  ParseParameters(config);

  update_timer_.SetRate(update_rate_);
  scan_publisher_ = nh_.advertise<sensor_msgs::LaserScan>(topic_, 1);
  galaxy_publisher_ = nh_.advertise<flatland_msgs::Galaxy2D>("galaxy2d_convex", 1);
  scan_vis_publisher_ = nh_.advertise<sensor_msgs::LaserScan>("scan_vis", 1);
  convex_polygon_vis_publisher_ = nh_.advertise<geometry_msgs::PolygonStamped>("convex_polygon_vis", 1);

  if_viz = false; // 定义一个变量来存储参数值
  nh_.getParam("/if_viz", if_viz); // 传递变量作为引用
  if_evenly_convex = false; // 定义一个变量来存储参数值
  nh_.getParam("/if_evenly_convex", if_evenly_convex); // 传递变量作为引用
  max_vertex_num = 120;
  nh_.getParam("/max_vertex_num", max_vertex_num); // 传递变量作为引用
  if_compute_convex = false;
  nh_.getParam("/if_compute_convex", if_compute_convex); // 传递变量作为引用


  // construct the body to laser transformation matrix once since it never
  // changes
  double c = cos(origin_.theta);
  double s = sin(origin_.theta);
  double x = origin_.x, y = origin_.y;
  m_body_to_laser_ << c, -s, x, s, c, y, 0, 0, 1;

  unsigned int num_laser_points =
      std::lround((max_angle_ - min_angle_) / increment_);

  // initialize size for the matrix storing the laser points
  m_laser_points_ = Eigen::MatrixXf(3, num_laser_points);
  m_world_laser_points_ = Eigen::MatrixXf(3, num_laser_points);
  v_zero_point_ << 0, 0, 1;

  // pre-calculate the laser points w.r.t to the laser frame, since this never
  // changes
  for (unsigned int i = 0; i < num_laser_points; i++) {
    float angle = min_angle_ + i * increment_;

    float x = range_ * cos(angle);
    float y = range_ * sin(angle);

    m_laser_points_(0, i) = x;
    m_laser_points_(1, i) = y;
    m_laser_points_(2, i) = 1;
  }

  // initialize constants in the laser scan message
  laser_scan_.angle_min = min_angle_;
  laser_scan_.angle_max = max_angle_;
  laser_scan_.angle_increment = increment_;
  laser_scan_.time_increment = 0;
  laser_scan_.scan_time = 0;
  laser_scan_.range_min = 0;
  laser_scan_.range_max = range_;
  laser_scan_.ranges.resize(num_laser_points);
  if (reflectance_layers_bits_)
    laser_scan_.intensities.resize(num_laser_points);
  else
    laser_scan_.intensities.resize(0);
  laser_scan_.header.seq = 0;
  laser_scan_.header.frame_id =
      tf::resolve("", GetModel()->NameSpaceTF(frame_id_));

  // Broadcast transform between the body and laser
  tf::Quaternion q;
  q.setRPY(0, 0, origin_.theta);

  laser_tf_.header.frame_id = tf::resolve(
      "", GetModel()->NameSpaceTF(body_->GetName()));  // Todo: parent_tf param
  laser_tf_.child_frame_id =
      tf::resolve("", GetModel()->NameSpaceTF(frame_id_));
  laser_tf_.transform.translation.x = origin_.x;
  laser_tf_.transform.translation.y = origin_.y;
  laser_tf_.transform.translation.z = 0;
  laser_tf_.transform.rotation.x = q.x();
  laser_tf_.transform.rotation.y = q.y();
  laser_tf_.transform.rotation.z = q.z();
  laser_tf_.transform.rotation.w = q.w();
}

void Laser::AfterPhysicsStep(const Timekeeper &timekeeper) {
  // keep the update rate
  if (!update_timer_.CheckUpdate(timekeeper)) {
    return;
  }

  // only compute and publish when the number of subscribers is not zero
  if (scan_publisher_.getNumSubscribers() > 0) {
    ComputeLaserRanges();
    laser_scan_.header.stamp = timekeeper.GetSimTime();

    Points scans_xy;
    Result3 res;
    flatland_msgs::Galaxy2D g2dres;

    // publish the laser scan
    scan_publisher_.publish(laser_scan_);
    
    if(if_compute_convex)
    {
      // convex publisher
      bool is_collision = process_scan_msg(laser_scan_,scans_xy);

      if(galaxy_xyin_360out(res, scans_xy,max_vertex_num,0,0,15.0,if_evenly_convex))
      {
        g2dres.success.data = true;
        g2dres.scans = laser_scan_.ranges;
        
        // polygon：它由一系列点组成，这些点定义了多边形的顶点。在这段代码中，polygon 的顶点来自 convex 向量，其中每个顶点由一个 geometry_msgs::Point32 类型的对象表示。Point32 对象包含三个浮点数字段 x, y, z，在这种情况下只使用 x 和 y 来表示二维空间中的点。
        // polar_convex、polar_convex_theta：极坐标下的距离、角度。相比于convex，polar_convex已经根据max_vertex_num计算重新Q分配后的点
        g2dres.convex_vertex = std::get<0>(res);
        g2dres.polar_convex = std::get<1>(res);
        g2dres.polar_convex_theta = std::get<2>(res);
      }
      else
      {
        g2dres.success.data = false;
        g2dres.scans = laser_scan_.ranges;
      }
      galaxy_publisher_.publish(g2dres); 

      if (if_viz)
      { 
        sensor_msgs::LaserScan scan_vis = laser_scan_;
        scan_vis.header.stamp = timekeeper.GetSimTime();
        // frame_id= nh_ + /robot

        std::string result = "sim_1/robot";
        // 找到第一个 '/' 的位置 from topic_
        size_t start = topic_.find_first_of('/');
        if (start != std::string::npos) {
            // 找到第二个 '/' 的位置
            size_t end = topic_.find('/', start + 1);
            if (end != std::string::npos) {
                // 提取从第一个 '/' 到第二个 '/' 之间的字符串（不包括这两个 '/'）
                std::string first_part = topic_.substr(start + 1, end - start - 1);
                // 组成新的字符串
                result = first_part + "/robot";
            }

        scan_vis.header.frame_id = result;
        scan_vis_publisher_.publish(scan_vis);

        geometry_msgs::PolygonStamped polygon_Stamped;
        polygon_Stamped.header.stamp = timekeeper.GetSimTime();
        polygon_Stamped.header.frame_id = result;
        polygon_Stamped.polygon = std::get<0>(res);
        convex_polygon_vis_publisher_.publish(polygon_Stamped);
        
        }
      }
    }
  }

  if (broadcast_tf_) {
    laser_tf_.header.stamp = timekeeper.GetSimTime();
    tf_broadcaster_.sendTransform(laser_tf_);
  }
}

// 
bool Laser::process_scan_msg(sensor_msgs::LaserScan& msg_LaserScan,Points& scans_xy) {
  bool is_collision = false;
  std::vector<float>& ranges = msg_LaserScan.ranges;
  size_t laser_num = ranges.size();

  if(laser_num < 360)
    std::cout<<"Laser.cpp: laser num wrong "<<laser_num << std::endl;
  double drad = 2*M_PI/laser_num;
  double laser_frame_theta = 0.,base_frame_x = 0.,base_frame_y = 0.;
  // // burger\originBot
  // double safe_dist = 0.113;//exp_dist + robot_radius
  // turtlebot2
  double safe_dist = 0.177;//exp_dist + robot_radius
  for(int i = 0;i < laser_num;++i)
  {
    if(std::isnan(ranges[i]))
      ranges[i] = msg_LaserScan.range_max;
    laser_frame_theta = i*drad + 0.;
    base_frame_x = ranges[i]*std::cos(laser_frame_theta) + 0.;
    base_frame_y = ranges[i]*std::sin(laser_frame_theta) + 0.;
    ranges[i] = std::hypot(base_frame_x,base_frame_y) - safe_dist;
    if(ranges[i] < 0.)
      is_collision = true;
    // scans_xy.emplace_back(ranges[i]*std::cos(laser_frame_theta),ranges[i]*std::sin(laser_frame_theta));
    scans_xy.emplace_back(base_frame_x,base_frame_y);
  }

  return is_collision;
}

void Laser::ComputeLaserRanges() {
  // get the transformation matrix from the world to the body, and get the
  // world to laser frame transformation matrix by multiplying the world to body
  // and body to laser
  const b2Transform &t = body_->GetPhysicsBody()->GetTransform();
  m_world_to_body_ << t.q.c, -t.q.s, t.p.x, t.q.s, t.q.c, t.p.y, 0, 0, 1;
  m_world_to_laser_ = m_world_to_body_ * m_body_to_laser_;

  // Get the laser points in the world frame by multiplying the laser points in
  // the laser frame to the transformation matrix from world to laser frame
  m_world_laser_points_ = m_world_to_laser_ * m_laser_points_;
  // Get the (0, 0) point in the laser frame
  v_world_laser_origin_ = m_world_to_laser_ * v_zero_point_;

  // Conver to Box2D data types
  b2Vec2 laser_origin_point(v_world_laser_origin_(0), v_world_laser_origin_(1));

  // Results vector
  std::vector<std::future<std::pair<double, double>>> results(
      laser_scan_.ranges.size());

  // loop through the laser points and call the Box2D world raycast by
  // enqueueing the callback
  for (unsigned int i = 0; i < laser_scan_.ranges.size(); ++i) {
    results[i] =
        pool_.enqueue([i, this, laser_origin_point] {  // Lambda function
          b2Vec2 laser_point;
          laser_point.x = m_world_laser_points_(0, i);
          laser_point.y = m_world_laser_points_(1, i);
          LaserCallback cb(this);

          GetModel()->GetPhysicsWorld()->RayCast(&cb, laser_origin_point,
                                                 laser_point);

          if (!cb.did_hit_) {
            return std::make_pair<double, double>(NAN, 0);
          } else {
            return std::make_pair<double, double>(cb.fraction_ * this->range_,
                                                  cb.intensity_);
          }
        });
  }

  // Unqueue all of the future'd results
  for (unsigned int i = 0; i < laser_scan_.ranges.size(); ++i) {
    auto result = results[i].get();  // Pull the result from the future
    laser_scan_.ranges[i] = result.first + this->noise_gen_(this->rng_);
    if (reflectance_layers_bits_) laser_scan_.intensities[i] = result.second;
  }
}

float LaserCallback::ReportFixture(b2Fixture *fixture, const b2Vec2 &point,
                                   const b2Vec2 &normal, float fraction) {
  uint16_t category_bits = fixture->GetFilterData().categoryBits;
  // only register hit in the specified layers
  if (!(category_bits & parent_->layers_bits_)) {
    return -1.0f;  // return -1 to ignore this hit
  }

  // Don't return on hitting sensors... they're not real
  if (fixture->IsSensor()) return -1.0f;

  if (category_bits & parent_->reflectance_layers_bits_) {
    intensity_ = 255.0;
  }

  did_hit_ = true;
  fraction_ = fraction;

  return fraction;
}

void Laser::ParseParameters(const YAML::Node &config) {
  // config: src/arena/simulation-setup/entities/robots/burger/yaml/burger.yaml
  YamlReader reader(config);
  std::string body_name = reader.Get<std::string>("body");
  topic_ = reader.Get<std::string>("topic", "scan");
  frame_id_ = reader.Get<std::string>("frame", GetName());
  broadcast_tf_ = reader.Get<bool>("broadcast_tf", true);
  update_rate_ = reader.Get<double>("update_rate",
                                    std::numeric_limits<double>::infinity());
  origin_ = reader.GetPose("origin", Pose(0, 0, 0));
  range_ = reader.Get<double>("range");
  noise_std_dev_ = reader.Get<double>("noise_std_dev", 0);

  std::vector<std::string> layers =
      reader.GetList<std::string>("layers", {"all"}, -1, -1);

  YamlReader angle_reader = reader.Subnode("angle", YamlReader::MAP);
  min_angle_ = angle_reader.Get<double>("min");
  max_angle_ = angle_reader.Get<double>("max");
  increment_ = angle_reader.Get<double>("increment");

  angle_reader.EnsureAccessedAllKeys();
  reader.EnsureAccessedAllKeys();

  if (max_angle_ < min_angle_) {
    throw YAMLException("Invalid \"angle\" params, must have max > min");
  }

  body_ = GetModel()->GetBody(body_name);
  if (!body_) {
    throw YAMLException("Cannot find body with name " + body_name);
  }

  std::vector<std::string> invalid_layers;
  layers_bits_ = GetModel()->GetCfr()->GetCategoryBits(layers, &invalid_layers);
  if (!invalid_layers.empty()) {
    throw YAMLException("Cannot find layer(s): {" +
                        boost::algorithm::join(invalid_layers, ",") + "}");
  }

  std::vector<std::string> reflectance_layer = {"reflectance"};
  reflectance_layers_bits_ =
      GetModel()->GetCfr()->GetCategoryBits(reflectance_layer, &invalid_layers);

  // init the random number generators
  std::random_device rd;
  rng_ = std::default_random_engine(rd());
  noise_gen_ = std::normal_distribution<double>(0.0, noise_std_dev_);

  ROS_DEBUG_NAMED("LaserPlugin",
                  "Laser %s params: topic(%s) body(%s, %p) origin(%f,%f,%f) "
                  "frame_id(%s) broadcast_tf(%d) update_rate(%f) range(%f)  "
                  "noise_std_dev(%f) angle_min(%f) angle_max(%f) "
                  "angle_increment(%f) layers(0x%u {%s})",
                  GetName().c_str(), topic_.c_str(), body_name.c_str(), body_,
                  origin_.x, origin_.y, origin_.theta, frame_id_.c_str(),
                  broadcast_tf_, update_rate_, range_, noise_std_dev_,
                  min_angle_, max_angle_, increment_, layers_bits_,
                  boost::algorithm::join(layers, ",").c_str());
}
};

PLUGINLIB_EXPORT_CLASS(flatland_plugins::Laser, flatland_server::ModelPlugin)
