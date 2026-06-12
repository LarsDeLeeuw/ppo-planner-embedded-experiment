cmake_minimum_required(VERSION 3.8)
project(tb3_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GridMap.msg"
  "msg/GridPose.msg"
  "msg/PowerSample.msg"
  "msg/PlannerMetrics.msg"
  "msg/ControlLoopStats.msg"
  "msg/ThermalStat.msg"
  "msg/BridgePredictTiming.msg"
  "msg/PredictTcpTiming.msg"
  "msg/ExperimentEvent.msg"
  "msg/MoveToGridResult.msg"
  "msg/RaplEnergy.msg"
  "msg/HostDiagnostics.msg"
  "msg/SessionMetadata.msg"
  "action/MoveToGrid.action"
  "srv/PredictAction.srv"
  "srv/SetGridPose.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
