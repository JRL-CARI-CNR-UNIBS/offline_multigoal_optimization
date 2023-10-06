cmake_minimum_required(VERSION 3.0.2)
project(multigoal_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  graph_core
  ik_solver_msgs
  roscpp
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES multigoal_solver
  CATKIN_DEPENDS geometry_msgs graph_core ik_solver_msgs roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/multigoal_solver.cpp
# )
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(compute_trees src/compute_trees.cpp)
add_dependencies(compute_trees ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(compute_trees
 ${catkin_LIBRARIES}
)


add_executable(compute_paths src/compute_paths.cpp)
add_dependencies(compute_paths ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(compute_paths
 ${catkin_LIBRARIES}
)

