#=============================================================================
# Finds the ROS library and components for integration into CMake projects.
#
# Simply add to your project:
#
#   set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/path/to/this/folder/")
#   find_package(ROS REQUIRED COMPONENTS roscpp tf2 cv_bridge pcl_ros)
#
# Then, link ros::<component> to your target
#   add_executable(myprogram main.cpp)
#   target_link_libraries(myprogram ros::roscpp ros::tf2 ros::cv_bridge ros::pcl_ros)
#
# If no COMPONENTS are specified, the default imported targets that will be generated are:
#   ros::roscpp ros::rostime ros::rosconsole ros::roscpp_serialization
#
# The following variables will also be defined:
#    ROS_FOUND
#    ROS_VERSION
#    ROS_INCLUDE_DIRS
#    ROS_LIBRARY_DIRS
#
# An alternative way to link ROS dependencies with these variables:
#   find_package(ROS REQUIRED)
#   target_link_directories(<target>
#     ${ROS_LIBRARY_DIRS}
#   )
#   target_include_directories(<target>
#     $<BUILD_INTERFACE:${ROS_INCLUDE_DIRS}>
#   )
#   target_link_libraries(<target>
#      roscpp rostime rosconsole roscpp_serialization tf
#   )
#
# However, with this method, cmake may not show any warnings/errors if the libraries
# you are linking to are not found.
# 

#=============================================================================
include(FindPackageHandleStandardArgs)

unset(ROS_FOUND)

# 1. Check if ROS_ROOT variable is set
if(DEFINED ENV{ROS_DISTRO} AND DEFINED ENV{ROS_ROOT})
  set(ROS_VERSION $ENV{ROS_DISTRO}) # Note the env variable ROS_VERSION is actually 1 for ROS 1.
  file(REAL_PATH $ENV{ROS_ROOT}/../.. ROS_ROOT) # Note the env variable ROS_ROOT actually points to <rosfolder>/share/ros
  set(ROS_FOUND TRUE)
elseif(DEFINED ENV{ROS_DISTRO}) # Find ROS location from $ROS_DISTRO
  set(ROS_VERSION $ENV{ROS_DISTRO})
  if(EXISTS /opt/ros/${ROS_VERSION})
    set(ROS_ROOT /opt/ros/${ROS_VERSION})
    set(ROS_FOUND TRUE)
  endif()
endif()
# Else find it in /opt/ros
if(NOT ROS_FOUND)
  # message(STATUS "ROS environment variable not set. Trying to find it...")
  set(_AVAILABLE_ROS_VERSIONS "noetic;melodic;lunar;kinetic;jade;indigo")
  foreach(_rosversion ${_AVAILABLE_ROS_VERSIONS})
    if(EXISTS "/opt/ros/${_rosversion}")
      set(ROS_FOUND TRUE)
      set(ROS_VERSION ${_rosversion})
      break()
    endif()
  endforeach()
  unset(_AVAILABLE_ROS_VERSIONS)
  set(ROS_ROOT /opt/ros/${ROS_VERSION})
endif()

# 2. Generate cmake targets ros::<package> to link to the libraries and their dependencies
# Also generate ROS_INCLUDE_DIRS and ROS_LIBRARY_DIRS for old school cmake people
if(ROS_FOUND)
  # Check if user-specified ROS_INCLUDE_DIR and ROS_LIBRARY_DIR is defined and exists
  if(NOT ROS_INCLUDE_DIR)
    set(ROS_INCLUDE_DIR ${ROS_ROOT}/include)
  endif()
  if(EXISTS "${ROS_INCLUDE_DIR}")
    set(ROS_INCLUDE_DIRS "${ROS_INCLUDE_DIR}")
  else()
    unset(ROS_INCLUDE_DIR)
  endif()

  find_path(ROS_LIBRARY_DIR
    libroscpp.so
    PATHS "${ROS_ROOT}/lib"
  )
  set(ROS_LIBRARY_DIRS "${ROS_LIBRARY_DIR}")

  # Set environment paths
  if(NOT CMAKE_PREFIX_PATH MATCHES "/opt/ros/${ROS_VERSION}")
    list(APPEND CMAKE_PREFIX_PATH "/opt/ros/${ROS_VERSION}")
  endif()
  # include(DashUtils)
  # DashAppendVarToEnv(LD_LIBRARY_PATH /opt/ros/noetic/lib)
  # if(${ROS_VERSION} STREQUAL "noetic")
  #   DashAppendVarToEnv(PYTHONPATH "/opt/ros/noetic/lib/python3/dist-packages")
  # else()
  #   DashAppendVarToEnv(PYTHONPATH "/opt/ros/noetic/lib/python2/dist-packages")
  # endif()
  # (/opt/ros/noetic/_setup_util.py does):
  # export CMAKE_PREFIX_PATH="/opt/ros/noetic"
  # export LD_LIBRARY_PATH="/opt/ros/noetic/lib"
  # export PATH="/opt/ros/noetic/bin:$PATH"
  # export PKG_CONFIG_PATH="/opt/ros/noetic/lib/pkgconfig"
  # export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"

  # Note: Using pkg-config to generate the imported targets is quite slow
  # and also result in strange breakages (glib/g_static failures) when 
  # multiple subdirectories try to link to the same PkgConfig::<target>

  # Default ros libraries to generate targets for.
  set(_ros_libs_to_find roscpp rostime rosconsole roscpp_serialization)
  if(ROS_FIND_COMPONENTS)
    list(APPEND _ros_libs_to_find ${ROS_FIND_COMPONENTS})
    list(REMOVE_DUPLICATES _ros_libs_to_find)
  endif()
  foreach(_ros_lib ${_ros_libs_to_find})
    find_package(${_ros_lib} QUIET PATHS "${ROS_ROOT}/share/${_ros_lib}/")
    if(${_ros_lib}_FOUND)
      set(ROS_${_ros_lib}_FOUND TRUE)
      list(APPEND ROS_GENERATED_TARGETS ros::${_ros_lib})
      if (NOT TARGET ros::${_ros_lib})
        add_library(${_ros_lib} INTERFACE)
        add_library(ros::${_ros_lib} ALIAS ${_ros_lib})
        target_include_directories(${_ros_lib} SYSTEM INTERFACE ${${_ros_lib}_INCLUDE_DIRS})
        target_link_directories(${_ros_lib} INTERFACE ${${_ros_lib}_LIBRARY_DIRS})
        target_link_libraries(${_ros_lib} INTERFACE ${${_ros_lib}_LIBRARIES})
      endif()
    else()
      message(WARNING "FindROS - Failed to find ${_ros_lib}")
    endif()
  endforeach()
  unset(_ros_libs_to_find)

  # #### Perform necessary features done in `source /opt/ros/<rosversion>/setup.sh` for people who are too lazy to source it
  # # Add /opt/ros/<rosversion> to CMAKE_PREFIX_PATH 
  # # IN_LIST seems to be broken for some cmake versions??
  # if (NOT CMAKE_PREFIX_PATH MATCHES "/opt/ros/${ROS_VERSION}")
  #   list(APPEND CMAKE_PREFIX_PATH "/opt/ros/${ROS_VERSION}")
  # endif()
  # # Add /opt/ros/<rosversion>/lib/python3/dist-packages to PYTHONPATH if it's not inside
  # set(_CURR_PYTHONPATH $ENV{PYTHONPATH})
  # string(REPLACE ":" ";" _CURR_PYTHONPATH_LIST "${_CURR_PYTHONPATH}") # Convert to CMake list form (semicolon separated)
  # if (NOT _CURR_PYTHONPATH_LIST MATCHES "/opt/ros/${ROS_VERSION}/lib/python3/dist-packages")
  #   list(PREPEND _CURR_PYTHONPATH_LIST /opt/ros/${ROS_VERSION}/lib/python3/dist-packages) # Prepend like setup.sh
  #   string(REPLACE ";" ":" _FIXED_PYTHONPATH "${_CURR_PYTHONPATH_LIST}") # Convert back to ENV form
  #   set(ENV{PYTHONPATH} ${_FIXED_PYTHONPATH})
  #   unset(_FIXED_PYTHONPATH)
  # endif()
  # unset(_CURR_PYTHONPATH)
  # unset(_CURR_PYTHONPATH_LIST)
  # ####

  # Useful printout
  # if (NOT ROS_FIND_QUIETLY)
  #   message(STATUS "FindROS:")
  #   message(STATUS "  ROS_VERSION: ${ROS_VERSION}")
  #   message(STATUS "  ROS_INCLUDE_DIRS: ${ROS_INCLUDE_DIRS}")
  #   message(STATUS "  ROS_LIBRARY_DIRS: ${ROS_LIBRARY_DIRS}")
  #   message(STATUS "  ROS_TARGETS: ${ROS_GENERATED_TARGETS}")
  # endif()
endif()

find_package_handle_standard_args(ROS
FOUND_VAR 
  ROS_FOUND
REQUIRED_VARS
  ROS_INCLUDE_DIR ROS_LIBRARY_DIR
VERSION_VAR
  ROS_VERSION
HANDLE_COMPONENTS
)

mark_as_advanced(ROS_INCLUDE_DIR ROS_LIBRARY_DIR ROS_VERSION)