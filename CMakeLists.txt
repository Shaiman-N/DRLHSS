cmake_minimum_required(VERSION 3.15)
project(cybersec_platform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/external)

add_executable(DRLHSS_main src/main.cpp
    src/sandboxPos_FP/Orchestrator_FP.cpp
    src/sandboxPos_FP/HostProfiler.cpp
    src/sandboxPos_FP/BaseImageSelector.cpp
    src/sandboxPos_FP/OverlayManager.cpp
    src/sandboxPos_FP/NamespaceManager.cpp
    src/sandboxPos_FP/SandboxRunner.cpp
    src/sandboxPos_FP/TelemetryCollector.cpp
    src/sandboxPos_FP/Cleaner.cpp
    src/sandboxNeg_FN/Orchestrator_FN.cpp
    src/sandboxNeg_FN/HostProfiler.cpp
    src/sandboxNeg_FN/BaseImageSelector.cpp
    src/sandboxNeg_FN/OverlayManager.cpp
    src/sandboxNeg_FN/NamespaceManager.cpp
    src/sandboxNeg_FN/SandboxRunner.cpp
    src/sandboxNeg_FN/TelemetryCollector.cpp
    src/sandboxNeg_FN/Cleaner.cpp
    )

target_include_directories(DRLHSS_main PUBLIC include)
target_link_libraries(DRLHSS_main pthread)
