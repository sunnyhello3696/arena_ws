#ifndef SIDEWAYS_INTER_H_
#define SIDEWAYS_INTER_H_

#include <ros/ros.h>
#include <mbf_costmap_core/costmap_inter.h>
#include <boost/thread/mutex.hpp>
#include <dynamic_reconfigure/server.h>
#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sideways_inter/sidewaysInterConfig.h>

#include <std_msgs/Float64.h>
#include <thread>



namespace sideways_inter
{

    class SidewaysInter : public mbf_costmap_core::CostmapInter
    {

        using mbf_costmap_core::CostmapInter::CostmapInter;

        /**
         * @brief Given a goal pose in the world, compute a plan
         * @param start The start pose
         * @param goal The goal pose
         * @param plan The plan... filled by the inter
         * @param cost The cost for the the plan
         * @param message Optional more detailed outcome as a string
         * @return Result code as described on GetInterPath action result:
         *         SUCCESS         = 0
         *         1..9 are reserved as plugin specific non-error results
         *         FAILURE         = 50  # Unspecified failure, only used for old, non-mfb_core based plugins
         *         CANCELED        = 51
         *         INVALID_START   = 52
         *         INVALID_GOAL    = 53
         *         BLOCKED_START   = 54
         *         BLOCKED_GOAL    = 55
         *         NO_PATH_FOUND   = 56
         *         PAT_EXCEEDED    = 57
         *         EMPTY_PATH      = 58
         *         TF_ERROR        = 59
         *         NOT_INITIALIZED = 60
         *         INVALID_PLUGIN  = 61
         *         INTERNAL_ERROR  = 62
         *         71..99 are reserved as plugin specific errors
         */
        uint32_t makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                          std::vector<geometry_msgs::PoseStamped> &plan, double &cost, std::string &message);

        /**
         * @brief  Set the plan that the planner is following
         * @param plan The plan to pass to the inter
         * @return True if the plan was updated successfully, false otherwise
         */
        bool setPlan(const std::vector<geometry_msgs::PoseStamped> &plan);

        /**
         * @brief Requests the inter to cancel, e.g. if it takes too much time.
         * @remark New on MBF API
         * @return True if a cancel has been successfully requested, false if not implemented.
         */
        bool cancel() { return false; };

        /**
         * @brief Initialization function for the CostmapInter
         * @param name The name of this inter
         * @param costmap_ros A pointer to the ROS wrapper of the costmap to use for planning
         */
        void initialize(std::string name, costmap_2d::Costmap2DROS *global_costmap_ros, costmap_2d::Costmap2DROS *local_costmap_ros);

        void resumeDriving(const ros::TimerEvent&);
    private:
        // mutexes
        boost::mutex plan_mtx_;
        boost::mutex speed_mtx_;

        // storage for setPlan
        std::vector<geometry_msgs::PoseStamped> plan_;

        // could be used for nh
        std::string name;
        std::string node_namespace_;
        double robot_radius_;


        ros::Timer wait_timer;
        ros::NodeHandle nh_;

        
        // default values
        // change in SidewaysInter.cfg to your preference
        double caution_detection_range_ = 10.0;
        double cautious_speed_ = 0.1;
        double ped_minimum_distance_ = 2.0;
        double temp_goal_distance_ = 2.0;
        double temp_goal_tolerance_ = 0.2;
        double fov_ = M_PI;
        double danger_threshold = 0.6;
        double distance;


        // variables to control the speed
        double speed_;
        double last_speed_;
        std::thread velocity_thread_;

        ros::Subscriber subscriber_;
        ros::Subscriber laser_scan_subscriber_;
        ros::Subscriber helios_points_subscriber_;

        ros::Publisher dangerPublisher; 

        ros::ServiceClient setParametersClient_;

        geometry_msgs::PoseStamped temp_goal_;
        bool new_goal_set_ = false;
        
        double max_vel_x_param_;
        double changed_max_vel_x_param_;

        dynamic_reconfigure::Reconfigure reconfig_;
        dynamic_reconfigure::DoubleParameter double_param_;
        dynamic_reconfigure::Config conf_;
        std::vector<geometry_msgs::Point32> semanticPoints;
        std::vector<double> detectedRanges;
        std::vector<double> detectedAngles;

        void reconfigure(sideways_inter::sidewaysInterConfig &config, uint32_t level);
        void semanticCallback(const crowdsim_msgs::SemanticData::ConstPtr& message);
        void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
        //void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
        void setMaxVelocityThread();
    };
}

#endif // SIDEWAYS_INTER_H_