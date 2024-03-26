cmake_minimum_required(VERSION 3.0.2)
project(bypass_inter)

find_package(catkin REQUIRED COMPONENTS
  mbf_costmap_core
)

# generate_dynamic_reconfigure_options(
#   cfg/BypassInter.cfg
# )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES bypass_inter
  CATKIN_DEPENDS mbf_costmap_core
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_library(bypass_inter src/bypass_inter.cpp)

add_dependencies(bypass_inter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(bypass_inter ${PROJECT_NAME}_gencfg)

target_link_libraries(bypass_inter
  ${catkin_LIBRARIES}
)


#############
## Install ##
#############

install(TARGETS bypass_inter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES mbf_inter_plugin.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


#############
## Testing ##
#############
