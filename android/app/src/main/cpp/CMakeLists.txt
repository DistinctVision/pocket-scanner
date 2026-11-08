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

find_program(GLSLC glslc
    PATHS "${ANDROID_NDK}/shader-tools/linux-x86_64"
          "$ENV{ANDROID_NDK_HOME}/shader-tools/linux-x86_64"
          "/home/alex/dev/android-sdk/ndk/28.2.13676358/shader-tools/linux-x86_64"
    NO_DEFAULT_PATH)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found; Vulkan shaders cannot be compiled")
endif()

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vk/shaders")
set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_HEADERS
    "${SHADER_OUT}/yuyv.comp.h"
    "${SHADER_OUT}/present.vert.h"
    "${SHADER_OUT}/present.frag.h")

add_custom_command(
    OUTPUT "${SHADER_OUT}/yuyv.comp.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT}"
    COMMAND ${GLSLC} --target-env=vulkan1.1 -mfmt=c -fshader-stage=compute
            "${SHADER_DIR}/yuyv.comp" -o "${SHADER_OUT}/yuyv.comp.h"
    DEPENDS "${SHADER_DIR}/yuyv.comp"
    VERBATIM)
add_custom_command(
    OUTPUT "${SHADER_OUT}/present.vert.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT}"
    COMMAND ${GLSLC} --target-env=vulkan1.1 -mfmt=c -fshader-stage=vert
            "${SHADER_DIR}/present.vert" -o "${SHADER_OUT}/present.vert.h"
    DEPENDS "${SHADER_DIR}/present.vert"
    VERBATIM)
add_custom_command(
    OUTPUT "${SHADER_OUT}/present.frag.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT}"
    COMMAND ${GLSLC} --target-env=vulkan1.1 -mfmt=c -fshader-stage=frag
            "${SHADER_DIR}/present.frag" -o "${SHADER_OUT}/present.frag.h"
    DEPENDS "${SHADER_DIR}/present.frag"
    VERBATIM)

find_program(SLANGC slangc
    PATHS "/home/alex/dev/slang/bin" "$ENV{HOME}/dev/slang/bin")
if(NOT SLANGC)
    message(FATAL_ERROR "slangc not found")
endif()

function(slang_shader source entry stage symbol)
    set(header "${SHADER_OUT}/${symbol}.h")
    set(spv "${SHADER_OUT}/${symbol}.spv")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT}"
        COMMAND ${SLANGC} "${source}" -target spirv -profile spirv_1_3 -stage ${stage} -entry ${entry}
                -matrix-layout-row-major -o "${spv}"
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/vk/spv_to_c.py" "${spv}" "${header}" "${symbol}"
        DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/vk/spv_to_c.py"
        VERBATIM)
endfunction()

slang_shader("${SHADER_DIR}/integrate.slang" integrate compute integrate_spv)
slang_shader("${SHADER_DIR}/marching.slang" clearDraw compute clear_draw_spv)
slang_shader("${SHADER_DIR}/marching.slang" clampDraw compute clamp_draw_spv)
slang_shader("${SHADER_DIR}/marching.slang" marching compute marching_spv)
slang_shader("${SHADER_DIR}/mesh.slang" meshVert vertex mesh_vert_spv)
slang_shader("${SHADER_DIR}/mesh.slang" meshFrag fragment mesh_frag_spv)
list(APPEND SHADER_HEADERS
    "${SHADER_OUT}/integrate_spv.h"
    "${SHADER_OUT}/clear_draw_spv.h"
    "${SHADER_OUT}/clamp_draw_spv.h"
    "${SHADER_OUT}/marching_spv.h"
    "${SHADER_OUT}/mesh_vert_spv.h"
    "${SHADER_OUT}/mesh_frag_spv.h")

add_library(pocket_vk SHARED
    vk/vk_device.cpp
    vk/vk_preview.cpp
    vk/preview_jni.cpp
    vk/volume.cpp
    ${SHADER_HEADERS})
set_source_files_properties(${SHADER_HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
target_include_directories(pocket_vk PRIVATE "${SHADER_OUT}")
target_compile_features(pocket_vk PRIVATE cxx_std_17)
target_link_libraries(pocket_vk vulkan android log)
