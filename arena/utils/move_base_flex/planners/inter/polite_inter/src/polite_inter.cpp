#include "../include/polite_inter.h"
#include "../../inter_util/include/inter_util.h"

#include <thread>
#include <std_msgs/Int32.h>

#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <ros/ros.h>
#include <ros/master.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>
#include <angles/angles.h>
#include <vector>

PLUGINLIB_EXPORT_CLASS(polite_inter::PoliteInter, mbf_costmap_core::CostmapInter)

namespace polite_inter
{

    uint32_t PoliteInter::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                                   std::vector<geometry_msgs::PoseStamped> &plan, double &cost, std::string &message)
    {
        boost::unique_lock<boost::mutex> plan_lock(plan_mtx_);
        boost::unique_lock<boost::mutex> speed_lock(speed_mtx_);

        double robot_x = start.pose.position.x;
        double robot_y = start.pose.position.y;
        double robot_z = start.pose.position.z;
        bool caution = false;
        bool wall_near = false;

        std::vector<double> distances;
        distances.empty();

        for (const auto &point : semanticPoints)
        {
            double distance = std::sqrt(std::pow(point.x - robot_x, 2) + std::pow(point.y - robot_y, 2)) + std::pow(point.z - robot_z, 2);
            distances.push_back(distance);

            // calculates if ped is behind the robot to determine if he can continue to drive or set temp_goal
            double angle_to_point = atan2(point.x - robot_x, point.y - robot_y);
            double theta = tf::getYaw(start.pose.orientation);
            double angle_diff = angles::shortest_angular_distance(theta, angle_to_point);

            for (size_t i = 0; i < detectedRanges.size(); ++i)
            {
                // check if scan could be pedestrian and if it is ignore it
                bool isPed = (detectedRanges[i] - 0.2 <= distance) && (distance <= detectedRanges[i] + 0.2);
                double relative_angle = angles::shortest_angular_distance(theta, detectedAngles[i]);
                // here we check if the scan is:
                // behind the robot, not a pedestrian and the temp goal would be in or behind the scanned object
                // to determine if the scan is a static obstacle
                if ((2 * std::abs(relative_angle) <= M_PI) && (detectedRanges[i] <= temp_goal_distance_) && !isPed)
                {
                    if (detectedRanges[i] <= 2*robot_radius_+0.07)
                    {
                        // TODO: Get robot size to determine appropiate value (currently hardcoded 0.6 for jackal)
                        ROS_INFO("Detected Range[%zu] that should be a static obstacle for Scan Point: %f and here the Angle %f", i, detectedRanges[i], 2 * std::abs(relative_angle));
                        wall_near = true;
                    }
                }
            }

            // check speed restriction
            caution |= (distance <= caution_detection_range_);

            // Check if the pedestrian is in range to set temporary goal and move back
            if ((!new_goal_set_) && (distance <= ped_minimum_distance_) && (2 * std::abs(angle_diff) <= fov_))
            {
                ROS_INFO("Pedestrian detected. Distance: %lf", distance);
                ROS_INFO("Setting new temp_goal");
                temp_goal_ = start;

                // calculating position for temporary goal
                temp_goal_.pose.position.x -= temp_goal_distance_ * cos(theta);
                temp_goal_.pose.position.y -= temp_goal_distance_ * sin(theta);
                temp_goal_.pose.orientation = tf::createQuaternionMsgFromYaw(tf::getYaw(temp_goal_.pose.orientation));
                temp_goal_.header.frame_id = start.header.frame_id;
                new_goal_set_ = true;
            }

            // nothing else to compute
            if (caution && new_goal_set_)
                break;
        }
        speed_ = caution ? changed_max_vel_x_param_ : max_vel_x_param_;
        inter_util::InterUtil::checkDanger(dangerPublisher, distances, 0.6);
        if (new_goal_set_)
        {
            if (wall_near)
            {
                ROS_ERROR("AVOIDED COLLISION WITH OBSTACLE. CONTINUE NORMAL PLANNING");
                new_goal_set_ = false;
                plan = plan_;
                return 0;
            }
            // calculate distance to temporary goal
            double distance_to_temp_goal = std::sqrt(std::pow(temp_goal_.pose.position.x - robot_x, 2) + std::pow(temp_goal_.pose.position.y - robot_y, 2));

            // Clear the existing plan and add temp_goal
            plan.clear();
            plan.push_back(temp_goal_);
            if (distance_to_temp_goal <= temp_goal_tolerance_)
            {
                ROS_INFO("Reached temp_goal. Resetting goal.");
                new_goal_set_ = false;
            }
        }
        else
            plan = plan_;

        return 0;
    }

    bool PoliteInter::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);

        plan_ = plan;
        return true;
    }

    void PoliteInter::semanticCallback(const crowdsim_msgs::SemanticData::ConstPtr &message)
    // turns our semantic layer data into points we can use to calculate distance
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

    void PoliteInter::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr &message)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);

        // Set a maximum distance threshold for wall detection (adjust as needed)
        double max_detection_range = caution_detection_range_ + 0.5; // detect every obstacle in his caution_detection_range plus 0.5 metres

        detectedRanges.clear();
        // Accessing and printing range data
        for (size_t i = 0; i < message->ranges.size(); ++i)
        {
            double angle = message->angle_min + i * message->angle_increment;
            double range = message->ranges[i];

            // Check if the range is under the maximum detection range
            if (range < max_detection_range)
            {
                detectedRanges.push_back(range);
                detectedAngles.push_back(angle);
            }
        }
    }

    void PoliteInter::initialize(std::string name, costmap_2d::Costmap2DROS *global_costmap_ros, costmap_2d::Costmap2DROS *local_costmap_ros)
    {
        this->name = name;
        std::string node_namespace_ = ros::this_node::getNamespace();
        std::string semantic_layer = "/crowdsim_agents/semantic/pedestrian";
        nh_ = ros::NodeHandle("~");
        subscriber_ = nh_.subscribe(semantic_layer, 1, &PoliteInter::semanticCallback, this);
        dangerPublisher = nh_.advertise<std_msgs::String>("Danger", 10);

        // get topic for our scan
        std::string scan_topic_name;
        std::string helios_points_topic_name;
        if (!nh_.getParam(node_namespace_ + "/move_base_flex/local_costmap/obstacles_layer/scan/topic", scan_topic_name))
        {
            ROS_ERROR("Failed to get parameter %s/move_base_flex/local_costmap/obstacles_layer/scan/topic", node_namespace_.c_str());
            if (!nh_.getParam(node_namespace_ + "/move_base_flex/local_costmap/obstacles_layer/helios_points/topic", helios_points_topic_name))
            {
                ROS_ERROR("Failed to get parameter %s/move_base_flex/local_costmap/obstacles_layer/helios_points/topic", node_namespace_.c_str());
            }
        }
        if (!nh_.getParam("/robot_radius", robot_radius_)){
            ROS_ERROR("Failed to get parameter %s/local_planner", node_namespace_.c_str());
        }
        if (!scan_topic_name.empty())
        {
            laser_scan_subscriber_ = nh_.subscribe(scan_topic_name, 1, &PoliteInter::laserScanCallback, this);
        }
        // if(!helios_points_topic_name.empty()){
        //     helios_points_subscriber_ = nh_.subscribe(helios_points_topic_name, 1, &PoliteInter::pointCloudCallback, this);
        // }

        // get our local planner name
        std::string planner_keyword;
        if (!nh_.getParam(node_namespace_ + "/local_planner", planner_keyword))
        {
            ROS_ERROR("Failed to get parameter %s/local_planner", node_namespace_.c_str());
        }
        std::string local_planner_name = inter_util::InterUtil::getLocalPlanner(planner_keyword);
        // get the starting parameter for max_vel_x from our planner
        if (!nh_.getParam(node_namespace_ + "/move_base_flex/" + local_planner_name + "/max_vel_x", max_vel_x_param_))
        {
            ROS_ERROR("Failed to get parameter %s/move_base_flex/%s/max_vel_x", node_namespace_.c_str(), local_planner_name.c_str());
            return;
        }
        // Create service client for the Reconfigure service
        setParametersClient_ = nh_.serviceClient<dynamic_reconfigure::Reconfigure>(node_namespace_ + "/move_base_flex/" + local_planner_name + "/set_parameters");
        dynamic_reconfigure::Server<polite_inter::PoliteInterConfig> server;
        server.setCallback(boost::bind(&PoliteInter::reconfigure, this, _1, _2));

        // thread to control the velocity for robot
        velocity_thread_ = std::thread(&PoliteInter::setMaxVelocityThread, this);

        // needs to be declared here because cautious_speed gets declared with reconfigure
        changed_max_vel_x_param_ = (cautious_speed_ * max_vel_x_param_);
    }

    void PoliteInter::reconfigure(polite_inter::PoliteInterConfig &config, uint32_t level)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);
        // updating values from config
        ped_minimum_distance_ = config.ped_minimum_distance;
        temp_goal_distance_ = config.temp_goal_distance;
        caution_detection_range_ = config.caution_detection_range;
        cautious_speed_ = config.cautious_speed;
        temp_goal_tolerance_ = config.temp_goal_tolerance;
        fov_ = config.fov;
        danger_threshold = config.danger_threshold;
        changed_max_vel_x_param_ = (cautious_speed_ * max_vel_x_param_);
    }

    void PoliteInter::setMaxVelocityThread()
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

}