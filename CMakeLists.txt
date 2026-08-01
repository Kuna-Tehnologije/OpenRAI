cmake_minimum_required(VERSION 3.16)
project(OpenRAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Include SQLite3 (bundled amalgamation)
include_directories(${PROJECT_SOURCE_DIR}/src/vendor/sqlite)

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
# SQLite3 amalgamation (compile as C)
set(SQLITE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor/sqlite/sqlite3.c")
set_source_files_properties(${SQLITE_SOURCE} PROPERTIES LANGUAGE C)

# Core sources (exclude main files and GUI variants)
file(GLOB CPP_SOURCES "src/*.cpp")
list(REMOVE_ITEM CPP_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main_chat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/chat_gui.cpp"        # Console version
    "${CMAKE_CURRENT_SOURCE_DIR}/src/chat_gui_win32.cpp"  # Win32 native GUI
    "${CMAKE_CURRENT_SOURCE_DIR}/src/chat_gui_imgui.cpp"  # ImGui version (future)
)

if(CUDA_ENABLED)
    file(GLOB CUDA_SOURCES "src/cuda/*.cu")

    # Main executable (original OpenRAI)
    add_executable(openrai src/main.cpp ${CPP_SOURCES} ${CUDA_SOURCES} ${SQLITE_SOURCE})
    set_target_properties(openrai PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(openrai PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(openrai PRIVATE Threads::Threads cudart cublas)

    # Chat executable - Console version
    add_executable(openrai_chat src/main_chat.cpp src/chat_gui.cpp ${CPP_SOURCES} ${CUDA_SOURCES} ${SQLITE_SOURCE})
    set_target_properties(openrai_chat PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(openrai_chat PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(openrai_chat PRIVATE Threads::Threads cudart cublas)

    # Chat GUI executable - Windows native UI (NO DEPENDENCIES!)
    add_executable(openrai_chat_gui WIN32 src/main_chat.cpp src/chat_gui_win32.cpp ${CPP_SOURCES} ${CUDA_SOURCES} ${SQLITE_SOURCE})
    set_target_properties(openrai_chat_gui PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(openrai_chat_gui PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(openrai_chat_gui PRIVATE Threads::Threads cudart cublas comctl32)
    target_compile_definitions(openrai_chat_gui PRIVATE OPENRAI_WIN32_GUI)
else()
    # Main executable (original OpenRAI)
    add_executable(openrai src/main.cpp ${CPP_SOURCES} ${SQLITE_SOURCE})
    target_link_libraries(openrai PRIVATE Threads::Threads)

    # Chat executable - Console version
    add_executable(openrai_chat src/main_chat.cpp src/chat_gui.cpp ${CPP_SOURCES} ${SQLITE_SOURCE})
    target_link_libraries(openrai_chat PRIVATE Threads::Threads)

    # Chat GUI executable - Windows native UI (NO DEPENDENCIES!)
    add_executable(openrai_chat_gui WIN32 src/main_chat.cpp src/chat_gui_win32.cpp ${CPP_SOURCES} ${SQLITE_SOURCE})
    target_link_libraries(openrai_chat_gui PRIVATE Threads::Threads comctl32)
    target_compile_definitions(openrai_chat_gui PRIVATE OPENRAI_WIN32_GUI)
endif()

