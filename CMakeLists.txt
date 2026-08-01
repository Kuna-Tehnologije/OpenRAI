cmake_minimum_required(VERSION 3.16)
project(OpenRAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------
# CUDA — auto-detected; set -DOPENRAI_CUDA=ON to force-enable
# -----------------------------------------------------------------------
option(OPENRAI_CUDA "Enable CUDA neural-network backend" ON)

if(OPENRAI_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA toolkit: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Detect GPU arch automatically via cmake 3.24+ or fall back
        if(POLICY CMP0104)
            cmake_policy(SET CMP0104 NEW)
        endif()

        # RTX 5070 = sm_120 (Blackwell), also include sm_86/89 as safe fallback
        set(CMAKE_CUDA_ARCHITECTURES "86;89;120" CACHE STRING "CUDA GPU architectures")

        add_compile_definitions(OPENRAI_CUDA_ENABLED)
        set(CUDA_ENABLED TRUE)
    else()
        message(STATUS "CUDA compiler not found - building CPU-only fallback")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    set(CUDA_ENABLED FALSE)
endif()

# -----------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------
file(GLOB CPP_SOURCES "src/*.cpp")

if(CUDA_ENABLED)
    file(GLOB CUDA_SOURCES "src/cuda/*.cu")
    add_executable(openrai ${CPP_SOURCES} ${CUDA_SOURCES})
    set_target_properties(openrai PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(openrai PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(openrai PRIVATE Threads::Threads cudart cublas)
else()
    add_executable(openrai ${CPP_SOURCES})
    target_link_libraries(openrai PRIVATE Threads::Threads)
endif()
