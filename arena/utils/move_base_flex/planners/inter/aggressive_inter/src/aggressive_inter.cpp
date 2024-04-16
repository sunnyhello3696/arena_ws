#include "../include/aggressive_inter.h"
#include "../../inter_util/include/inter_util.h"

#include <thread>
#include <vector>

#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <ros/ros.h>
#include <ros/master.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>
PLUGINLIB_EXPORT_CLASS(aggressive_inter::AggressiveInter, mbf_costmap_core::CostmapInter)

namespace aggressive_inter
{
    uint32_t AggressiveInter::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                                       std::vector<geometry_msgs::PoseStamped> &plan, double &cost, std::string &message)
    {
        boost::unique_lock<boost::mutex> plan_lock(plan_mtx_);
        boost::unique_lock<boost::mutex> speed_lock(speed_mtx_);

        double robot_x = start.pose.position.x;
        double robot_y = start.pose.position.y;
        double robot_z = start.pose.position.z;
        double minDistance = INFINITY;

        std::vector<double> distances;
        distances.empty();
        for (const auto &point : semanticPoints)
        {
            double distance = std::sqrt(std::pow(point.x - robot_x, 2) + std::pow(point.y - robot_y, 2)) + std::pow(point.z - robot_z, 2);
            minDistance = std::min(minDistance, distance);
            distances.push_back(distance);
        }
        inter_util::InterUtil::checkDanger(dangerPublisher, distances, danger_threshold);
        ROS_WARN("Danger level: %f", inter_util::InterUtil::getDangerLevel(distances));
        ROS_WARN("Distances: ");
        for (const double& distance : distances) {
            ROS_WARN("%f", distance);
        }

        double temp_speed = max_vel_x_param_;
        
        // Check if the closest pedestrian is in range to slow down
        if (minDistance <= slowdown_distance)
        {
            //speed converges to max_vel_x_param_ at around 5 -> adapt function if necessary
            temp_speed = max_vel_x_param_ - (max_vel_x_param_ / (1 + std::pow(minDistance, 2)));
        }

        speed_ = temp_speed;
        plan = plan_;
        return 0;
    }

    bool AggressiveInter::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        plan_ = plan;
        return true;
    }

    void AggressiveInter::semanticCallback(const crowdsim_msgs::SemanticData::ConstPtr &message)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);
        
        semanticPoints.clear();
        for (const auto &point : message->points)
        {
            geometry_msgs::Point32 pedestrianPoint;
            pedestrianPoint.x = point.location.x;
            pedestrianPoint.y = point.location.y;
            pedestrianPoint.z = point.location.z;
            semanticPoints.push_back(pedestrianPoint);
        }
    }

    void AggressiveInter::initialize(std::string name, costmap_2d::Costmap2DROS *global_costmap_ros, costmap_2d::Costmap2DROS *local_costmap_ros)
    {
        this->name = name;
        std::string node_namespace_ = ros::this_node::getNamespace();
        std::string semantic_layer = "/crowdsim_agents/semantic/pedestrian";
        nh_ = ros::NodeHandle("~");
        dangerPublisher = nh_.advertise<std_msgs::String>("Danger", 10);  
        subscriber_ = nh_.subscribe(semantic_layer, 1, &AggressiveInter::semanticCallback, this);
        // get our local planner name
        std::string planner_keyword;
        if (!nh_.getParam(node_namespace_+"/local_planner", planner_keyword)){
            ROS_ERROR("Failed to get parameter %s/local_planner", node_namespace_.c_str());
        }
        std::string local_planner_name = inter_util::InterUtil::getLocalPlanner(planner_keyword);
        // get the starting parameter for max_vel_x from our planner
        if (!nh_.getParam(node_namespace_+"/move_base_flex/"+ local_planner_name +"/max_vel_x", max_vel_x_param_))
        {
            ROS_ERROR("Failed to get parameter %s/move_base_flex/%s/max_vel_x", node_namespace_.c_str(), local_planner_name.c_str());
            return;
        }
        // Create service clients for the GetDump and Reconfigure services
        setParametersClient_ = nh_.serviceClient<dynamic_reconfigure::Reconfigure>(node_namespace_+"/move_base_flex/"+ local_planner_name+"/set_parameters");
        dynamic_reconfigure::Server<aggressive_inter::AggressiveInterConfig> server;
        server.setCallback(boost::bind(&AggressiveInter::reconfigure, this, _1, _2));

        // thread to control the velocity for robot
        velocity_thread_ = std::thread(&AggressiveInter::setMaxVelocityThread, this);
    }

    void AggressiveInter::setMaxVelocityThread()
    {
        ros::Rate rate(1); // Adjust the rate as needed
        while (ros::ok())
        {
            // Lock to access shared variables
            boost::unique_lock<boost::mutex> lock(speed_mtx_);

            // Check if the speed has changed
            if (speed_ != last_speed_)
            {
                // set max_vel_x parameter
                double_param_.name = "max_vel_x";
                double_param_.value = speed_;
                conf_.doubles.clear();
                conf_.doubles.push_back(double_param_);
                reconfig_.request.config = conf_;

                // Call setParametersClient_ to update parameters
                if (setParametersClient_.call(reconfig_))
                {
                    ROS_INFO_ONCE("Dynamic reconfigure request successful");
                }
                else
                {
                    ROS_ERROR_ONCE("Failed to call dynamic reconfigure service");
                }

                // Update last_speed_ to avoid unnecessary calls
                last_speed_ = speed_;
            }

            // Unlock and sleep
            lock.unlock();
            rate.sleep();
        }
    }

    void AggressiveInter::reconfigure(aggressive_inter::AggressiveInterConfig &config, uint32_t level)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);

        slowdown_distance = config.slowdown_distance;
        danger_threshold = config.danger_threshold;
    }
}