#ifndef SHORTSIGHTED_INTER_H_
#define SHORTSIGHTED_INTER_H_

#include <ros/ros.h>
#include <mbf_costmap_core/costmap_inter.h>
#include <boost/thread/mutex.hpp>

#include <dynamic_reconfigure/server.h>
#include <shortsighted_inter/ShortsightedInterConfig.h>

namespace shortsighted_inter
{

    class ShortsightedInter : public mbf_costmap_core::CostmapInter
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

    private:
        // storage for setPlan
        std::vector<geometry_msgs::PoseStamped> plan_;
        boost::mutex plan_mtx_;

        // could be used for nh
        std::string name;

        // max vision
        double vision_limit_ = 0.2;

        // min poses in path
        size_t min_poses_ = 1;

        boost::mutex vision_cfg_mtx_;

        void reconfigure(shortsighted_inter::ShortsightedInterConfig &config, uint32_t level);
    };
}

#endif // SHORTSIGHTED_INTER_H_