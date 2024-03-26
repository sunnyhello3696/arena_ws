#ifndef ROS_PARAMS

#include <string>

#define ROS_PARAMS
struct ROSParams {
    std::string robotNamespace;
    int targetState; 
};

#endif