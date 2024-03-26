cmake_minimum_required(VERSION 3.0.2)
project(aggressive_inter)

find_package(catkin REQUIRED COMPONENTS
  mbf_costmap_core
  inter_util
)

generate_dynamic_reconfigure_options(
  cfg/AggressiveInter.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES aggressive_inter
  CATKIN_DEPENDS mbf_costmap_core inter_util
)

# ##########
# # Build ##
# ##########
include_directories(

  # include
  ${catkin_INCLUDE_DIRS}
)

add_library(aggressive_inter src/aggressive_inter.cpp)

add_dependencies(aggressive_inter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(aggressive_inter ${PROJECT_NAME}_gencfg)

target_link_libraries(aggressive_inter
  ${catkin_LIBRARIES}
)

# ############
# # Install ##
# ############
install(TARGETS aggressive_inter
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

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# ############
# # Testing ##
# ############