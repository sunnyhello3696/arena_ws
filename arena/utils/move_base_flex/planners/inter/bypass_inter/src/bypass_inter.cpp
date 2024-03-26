#include "../include/bypass_inter.h"
#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(bypass_inter::BypassInter, mbf_costmap_core::CostmapInter)

namespace bypass_inter
{

    uint32_t BypassInter::makePlan(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal,
                                   std::vector<geometry_msgs::PoseStamped> &plan, double &cost, std::string &message)
    {
        plan = plan_;
        return 0;
    }

    bool BypassInter::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        plan_ = plan;
        return true;
    }

    void BypassInter::initialize(std::string name, costmap_2d::Costmap2DROS *global_costmap_ros, costmap_2d::Costmap2DROS *local_costmap_ros){
        this->name = name;
        //doesn't need any of the costmaps here
    }

}