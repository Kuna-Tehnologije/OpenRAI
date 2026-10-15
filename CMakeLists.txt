cmake_minimum_required(VERSION 3.16)
project(OpenRAI LANGUAGES C CXX)

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

        # Imported CUDA::cudart target (CMake >= 3.17); plain "cudart" otherwise
        find_package(CUDAToolkit QUIET)
    else()
        message(STATUS "CUDA compiler not found - building CPU-only fallback")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    set(CUDA_ENABLED FALSE)
endif()

# -----------------------------------------------------------------------
# Sources
#   openrai_core : everything except main.cpp (shared by the engine binary
#                  and the test binary)
# -----------------------------------------------------------------------
file(GLOB CPP_SOURCES "src/*.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/src/main\\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/src/main_chat\\.cpp$")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/src/chat_gui.*\\.cpp$")

# Bundled SQLite amalgamation (learning store).  Compiled as C.
set(SQLITE_SOURCE "${PROJECT_SOURCE_DIR}/src/vendor/sqlite/sqlite3.c")
set_source_files_properties(${SQLITE_SOURCE} PROPERTIES LANGUAGE C)
set(SQLITE_DEFS SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION SQLITE_DQS=0)
list(APPEND CPP_SOURCES ${SQLITE_SOURCE})

if(CUDA_ENABLED)
    file(GLOB CUDA_SOURCES "src/cuda/*.cu")
    add_library(openrai_core STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
    set_target_properties(openrai_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(openrai_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/vendor/sqlite
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_compile_definitions(openrai_core PRIVATE ${SQLITE_DEFS})
    if(TARGET CUDA::cudart)
        target_link_libraries(openrai_core PUBLIC Threads::Threads CUDA::cudart)
    else()
        target_link_libraries(openrai_core PUBLIC Threads::Threads cudart)
    endif()
else()
    add_library(openrai_core STATIC ${CPP_SOURCES})
    target_include_directories(openrai_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/vendor/sqlite)
    target_compile_definitions(openrai_core PRIVATE ${SQLITE_DEFS})
    target_link_libraries(openrai_core PUBLIC Threads::Threads)
    if(UNIX AND NOT APPLE)
        target_link_libraries(openrai_core PUBLIC ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(openrai src/main.cpp)
target_link_libraries(openrai PRIVATE openrai_core)

# -----------------------------------------------------------------------
# Tests  (cmake -DOPENRAI_BUILD_TESTS=ON ..  &&  ctest  or  ./openrai_tests)
# -----------------------------------------------------------------------
option(OPENRAI_BUILD_TESTS "Build the node scheduler tests" ON)
if(OPENRAI_BUILD_TESTS)
    enable_testing()
    add_executable(openrai_tests tests/node_tests.cpp)
    target_link_libraries(openrai_tests PRIVATE openrai_core)
    add_test(NAME node_scheduler COMMAND openrai_tests)
endif()

# -----------------------------------------------------------------------
# Optional Dear ImGui front-end  (text in / answer out + live node panel)
#   cmake -DOPENRAI_IMGUI_DIR=/path/to/imgui ..      (needs GLFW + OpenGL)
# -----------------------------------------------------------------------
set(OPENRAI_IMGUI_DIR "" CACHE PATH "Path to a Dear ImGui checkout; enables the openrai_gui target")
if(OPENRAI_IMGUI_DIR)
    if(NOT EXISTS "${OPENRAI_IMGUI_DIR}/imgui.cpp")
        message(FATAL_ERROR "OPENRAI_IMGUI_DIR=${OPENRAI_IMGUI_DIR} does not contain imgui.cpp")
    endif()
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    add_executable(openrai_gui
        ui/imgui_frontend.cpp
        ${OPENRAI_IMGUI_DIR}/imgui.cpp
        ${OPENRAI_IMGUI_DIR}/imgui_draw.cpp
        ${OPENRAI_IMGUI_DIR}/imgui_tables.cpp
        ${OPENRAI_IMGUI_DIR}/imgui_widgets.cpp
        ${OPENRAI_IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${OPENRAI_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(openrai_gui PRIVATE ${OPENRAI_IMGUI_DIR} ${OPENRAI_IMGUI_DIR}/backends)
    target_link_libraries(openrai_gui PRIVATE openrai_core glfw OpenGL::GL)
    message(STATUS "openrai_gui enabled (ImGui at ${OPENRAI_IMGUI_DIR})")
endif()

# -----------------------------------------------------------------------
# Chat front-ends (ported from the feature/language-model branch)
#   openrai_chat        console chat, every platform
#   openrai_chat_gui    native Win32 window, no dependencies (Windows only)
#   openrai_chat_imgui  ImGui + DirectX 11 (Windows, needs vcpkg imgui with
#                       win32-binding + dx11-binding)
# -----------------------------------------------------------------------
add_executable(openrai_chat src/main_chat.cpp src/chat_gui.cpp)
target_link_libraries(openrai_chat PRIVATE openrai_core)

if(WIN32)
    add_executable(openrai_chat_gui WIN32 src/main_chat.cpp src/chat_gui_win32.cpp)
    target_link_libraries(openrai_chat_gui PRIVATE openrai_core comctl32)
    target_compile_definitions(openrai_chat_gui PRIVATE OPENRAI_WIN32_GUI)

    find_package(imgui CONFIG QUIET)
    if(imgui_FOUND)
        add_executable(openrai_chat_imgui WIN32 src/main_chat.cpp src/chat_gui_imgui.cpp)
        target_link_libraries(openrai_chat_imgui PRIVATE openrai_core imgui::imgui d3d11 dxgi)
        target_compile_definitions(openrai_chat_imgui PRIVATE OPENRAI_USE_IMGUI)
        message(STATUS "openrai_chat_imgui enabled (vcpkg imgui found)")
    endif()
endif()
