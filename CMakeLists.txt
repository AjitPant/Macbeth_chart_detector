set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package(OpenCV  REQUIRED )
message(STATUS "include dir: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( demo src/demo.cpp src/utils.cpp )
target_link_libraries( demo ${OpenCV_LIBS} )
