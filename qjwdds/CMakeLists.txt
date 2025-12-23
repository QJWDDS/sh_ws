cmake_minimum_required(VERSION 3.8)
project(qjwdds)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)#!!!!!!!!!!!!
find_package(px4_msgs REQUIRED)

# 生成自定义消息
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImageDeviation.msg"
)

#================
install(DIRECTORY launch                             
  DESTINATION share/${PROJECT_NAME}
)

#================
# 添加可执行文件
# 添加依赖：
# 安装可执行文件
#======================================
add_executable(test_c src/control/test_c.cpp)  
ament_target_dependencies(test_c rclcpp px4_msgs)  
target_link_libraries(test_c                        
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
install(TARGETS test_c                             
  DESTINATION lib/${PROJECT_NAME}
)
#======================================
add_executable(sh_vision_control_s src/control/sh_vision_control_s.cpp)  
ament_target_dependencies(sh_vision_control_s rclcpp px4_msgs)  
target_link_libraries(sh_vision_control_s                        
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
install(TARGETS sh_vision_control_s                             
  DESTINATION lib/${PROJECT_NAME}
)
#======================================
add_executable(sh_vision_control src/control/sh_vision_control.cpp)  
ament_target_dependencies(sh_vision_control rclcpp px4_msgs)  
target_link_libraries(sh_vision_control                        
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
install(TARGETS sh_vision_control                             
  DESTINATION lib/${PROJECT_NAME}
)
#======================================
add_executable(virtual_control src/control/virtual_control.cpp)  
ament_target_dependencies(virtual_control rclcpp px4_msgs)  
target_link_libraries(virtual_control                        
  ${PROJECT_NAME}__rosidl_typesupport_cpp
)
install(TARGETS virtual_control                             
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
