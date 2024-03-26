#include "../include/shortsighted_inter.h"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(shortsighted_inter::ShortsightedInter, mbf_costmap_core::CostmapInter)

namespace shortsighted_inter
{

    uint32_t ShortsightedInter::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                                   std::vector<geometry_msgs::PoseStamped> &plan, double &cost, std::string &message)
    {
        

        boost::unique_lock<boost::mutex> lock(vision_cfg_mtx_);
        boost::unique_lock<boost::mutex> lock2(plan_mtx_);

        size_t limit = std::floor(plan_.size() * vision_limit_);
        limit = std::max(limit, min_poses_);
        limit = std::min(limit, plan_.size());

        plan = std::vector<geometry_msgs::PoseStamped>(plan_.begin(), plan_.begin() + limit);

        return 0;
    }

    bool ShortsightedInter::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        boost::unique_lock<boost::mutex> lock(plan_mtx_);
        plan_ = plan;
        return true;
    }

    void ShortsightedInter::initialize(std::string name, costmap_2d::Costmap2DROS *global_costmap_ros, costmap_2d::Costmap2DROS *local_costmap_ros){
        this->name = name;

        dynamic_reconfigure::Server<shortsighted_inter::ShortsightedInterConfig> server;
        server.setCallback(boost::bind(&ShortsightedInter::reconfigure, this, _1, _2));
    }

    void ShortsightedInter::reconfigure(shortsighted_inter::ShortsightedInterConfig &config, uint32_t level)
    {
        boost::unique_lock<boost::mutex> lock(vision_cfg_mtx_);
        vision_limit_ = config.vision_limit;
        min_poses_ = config.min_poses;
    }

}