cmake_minimum_required (VERSION 2.8.11)

project(caffefication)

# ------------------------------------------------------------------------------
# C++11 settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()


set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES -DNO_STRICT)

# ------------------------------------------------------------------------------
# Caffe
# ------------------------------------------------------------------------------
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

add_definitions(${Caffe_DEFINITIONS})
list(APPEND LIBRARIES ${Caffe_LIBRARIES})

message(STATUS "Caffe_DEFINITIONS")
message(STATUS ${Caffe_DEFINITIONS})

message(STATUS "Caffe_INCLUDE_DIRS")
message(STATUS ${Caffe_INCLUDE_DIRS})

message(STATUS "Caffe_LIBRARIES")
message(STATUS ${Caffe_LIBRARIES})

add_executable(nfdd-c3d nfdd-c3d.cpp)
target_link_libraries(nfdd-c3d ${LIBRARIES})

add_executable(nfdd-c3d-main nfdd-c3d-main.cpp)
target_link_libraries(nfdd-c3d-main ${LIBRARIES})

add_executable(nfdd-c3d-main4nist nfdd-c3d-main4nist.cpp)
target_link_libraries(nfdd-c3d-main4nist ${LIBRARIES})

add_executable(test-dryrun nfdd-c3d-dryrun.cpp test-dryrun.cpp)
target_link_libraries(test-dryrun ${LIBRARIES})
