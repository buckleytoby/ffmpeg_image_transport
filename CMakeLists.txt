cmake_minimum_required(VERSION 3.5)

project(ffmpeg_image_transport)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# the image transport api changed between distros
if(DEFINED ENV{ROS_DISTRO})
  if($ENV{ROS_DISTRO} STREQUAL "foxy" OR
      $ENV{ROS_DISTRO} STREQUAL "galactic")
    add_definitions(-DUSE_OLD_IMAGE_TRANSPORT_API)
  endif()
else()
  message(ERROR "ROS_DISTRO environment variable is not set!")
endif()


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcutils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ffmpeg_image_transport_msgs REQUIRED)

find_package(OpenCV REQUIRED imgproc)

#set(FFMPEG_PKGCONFIG "/home/nuc-haptics/ffmpeg_build/lib/pkgconfig" CACHE STRING "ffmpeg_pkgconfig string")
#set(CMAKE_INSTALL_RPATH "/home/nuc-haptics/ffmpeg_build/lib")
set(FFMPEG_PKGCONFIG "" CACHE STRING "ffmpeg_pkgconfig string")
if("${FFMPEG_PKGCONFIG}" STREQUAL "")
else()
  message(WARNING "using FFMPEG package from ${FFMPEG_PKGCONFIG}")
endif()

set(ENV{PKG_CONFIG_PATH} ${FFMPEG_PKGCONFIG})
message("env pkg_config_path: $ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libswresample
    libswscale
    libavutil)

set(LIBRARY_NAME ${PROJECT_NAME}_component)



###### Cuda ######
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
set(CUDA_LIBRARIES "libcuda.so")
enable_language(CUDA)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")




add_library(cuda_encoder SHARED src/cuda_encoder.cu)
target_include_directories(cuda_encoder PRIVATE include)
target_include_directories(cuda_encoder PRIVATE /usr/include/)
target_compile_options(cuda_encoder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --disable-warnings
                       >)

install(TARGETS cuda_encoder EXPORT export_cuda_encoder
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
ament_export_targets(export_cuda_encoder)

###### END Cuda ######

###### CUSTOM FFMPEG #####
set(CUSTOM_FFMPEG_INCLUDES ~/ffmpeg_build/include)
set(CUSTOM_FFMPEG_LIBS ~/ffmpeg_build/lib)

###### END CUSTOM FFMPEG #####

add_library(
  ${LIBRARY_NAME}
  SHARED
  src/ffmpeg_publisher.cpp
  src/ffmpeg_subscriber.cpp
  src/ffmpeg_encoder.cpp
  src/ffmpeg_decoder.cpp
  src/tdiff.cpp
  src/manifest.cpp)

target_include_directories(${LIBRARY_NAME} PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
  /usr/include/)

ament_target_dependencies(${LIBRARY_NAME}
  cv_bridge
  image_transport
  pluginlib
  rclcpp
  sensor_msgs
  ffmpeg_image_transport_msgs)

target_link_libraries(${LIBRARY_NAME} opencv_imgproc PkgConfig::LIBAV cuda_encoder ${CUDA_LIBRARIES})

ament_export_dependencies(
  OpenCV
  cv_bridge
  image_transport
  pluginlib
  rclcpp
  rcutils
  sensor_msgs
  std_msgs
  ffmpeg_image_transport_msgs)

install(TARGETS ${LIBRARY_NAME} EXPORT export_${LIBRARY_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)


ament_export_targets(export_${LIBRARY_NAME})

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME})

pluginlib_export_plugin_description_file(image_transport ffmpeg_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_cpplint REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_pep257 REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  ament_copyright()
  ament_cppcheck(LANGUAGE c++)
  ament_cpplint(FILTERS "-build/include,-runtime/indentation_namespace")
  ament_flake8()
  ament_lint_cmake()
  ament_pep257()
  ament_clang_format(CONFIG_FILE .clang-format)
  ament_xmllint()
endif()

ament_package()
