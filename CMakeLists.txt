cmake_minimum_required(VERSION 2.8)
project(PD-Flow)

# Set optimized building:
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
endif(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

include_directories(include)

set(cuda_src src/pdflow_cudalib.cu)

################
# Core library #
################

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# set(CUDA_NVCC_FLAGS "-arch=sm_20")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")

# cuda_add_library(pdflow_cudalib ${cuda_src} SHARED)

###################
# Library wrapper #
###################

# find_package(OpenCV REQUIRED)
cuda_compile(cuda_objs ${cuda_src})
add_library(scene_flow SHARED ${cuda_src} ${cuda_objs} src/scene_flow.cpp)
target_link_libraries(scene_flow ${CUDA_LIBRARIES})
# add_library(scene_flow SHARED ${cuda_src} ${cuda_objs} src/scene_flow.cpp)
# target_link_libraries(scene_flow ${OpenCV_LIBS} ${CUDA_LIBRARIES})
# add_library(scene_flow SHARED src/scene_flow.cpp)
# target_link_libraries(scene_flow ${OpenCV_LIBS} ${CUDA_LIBRARIES} pdflow_cudalib)

#################
# Bundled demos #
#################

# SET(DEFAULT_BUILD_EVALUATOR ON)
# SET(BUILD_EVALUATOR ${DEFAULT_BUILD_EVALUATOR} CACHE BOOL "Build the scene flow evaluator for a RGB-D frame pair that uses OpenCV")

# #Build the scene flow evaluator
# IF (BUILD_EVALUATOR)
# 	FIND_PACKAGE(OpenCV REQUIRED)
	
# 	ADD_EXECUTABLE(Scene-Flow-Impair
# 		src/apps/main_scene_flow_impair.cpp
# 		src/apps/scene_flow_impair.cpp
# 		src/apps/scene_flow_impair.h)
		
# 	TARGET_LINK_LIBRARIES(Scene-Flow-Impair 
# 			${OpenCV_LIBS}
# 			${CUDA_LIBRARIES}
# 			pdflow_cudalib)
# ENDIF (BUILD_EVALUATOR)

# SET(DEFAULT_BUILD_RT_VISUALIZATION ON)
# SET(BUILD_RT_VISUALIZATION ${DEFAULT_BUILD_RT_VISUALIZATION} CACHE BOOL "Build the scene flow estimator with real-time visualization that requires MRPT and OpenNI2")

# #Build real-time scene flow with visualization
# IF (BUILD_RT_VISUALIZATION)
# 	FIND_PACKAGE(MRPT REQUIRED base gui opengl)

# 	IF (WIN32)
# 		INCLUDE_DIRECTORIES($ENV{OPENNI2_INCLUDE})
# 		LINK_DIRECTORIES($ENV{OPENNI2_LIB})
# 		SET(OpenNI_lib "$ENV{OPENNI2_LIB}/OpenNI2.lib")
# 	ENDIF (WIN32)
	
# 	IF (UNIX)
# 		SET(OpenNI2_libdir "/usr/lib")
# 		INCLUDE_DIRECTORIES("/usr/include/openni2")
# 		LINK_DIRECTORIES(${OpenNI2_libdir})
# 		SET(OpenNI_lib "${OpenNI2_libdir}/libOpenNI2.so")
# 	ENDIF (UNIX)

# 	ADD_EXECUTABLE(Scene-Flow-Visualization
# 		src/apps/main_scene_flow_visualization.cpp
# 		src/apps/scene_flow_visualization.cpp
# 		src/apps/scene_flow_visualization.h
# 		src/apps/legend_pdflow.xpm)
		
# 	TARGET_LINK_LIBRARIES(Scene-Flow-Visualization
# 			${MRPT_LIBS}
# 			${CUDA_LIBRARIES}
# 			${OpenNI_lib}
# 			pdflow_cudalib)
# ENDIF (BUILD_RT_VISUALIZATION)
