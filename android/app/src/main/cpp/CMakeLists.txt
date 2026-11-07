cmake_minimum_required(VERSION 3.22)
project(pocket_slam)

if(NOT STELLA_PREFIX)
    set(STELLA_PREFIX "/home/alex/dev/android-prefix/arm64")
endif()
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "${STELLA_PREFIX}/sdk/native/jni")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Link the stripped copies. They live outside jniLibs: AGP packages a shared
# library it links, and jniLibs holds only the libraries this .so does not name.
set(SLAM_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../slam-libs/${ANDROID_ABI}")

add_library(pocket_slam SHARED slam_jni.cpp)
target_include_directories(pocket_slam PRIVATE
    "${STELLA_PREFIX}/include"
    "${STELLA_PREFIX}/include/eigen3"
    "${STELLA_PREFIX}/include/stella_vslam/3rd/json/include"
    ${OpenCV_INCLUDE_DIRS})
target_compile_features(pocket_slam PRIVATE cxx_std_17)
target_link_libraries(pocket_slam
    "${SLAM_LIBS}/libstella_vslam.so"
    "${SLAM_LIBS}/libopencv_imgproc.so"
    "${SLAM_LIBS}/libopencv_core.so"
    log)
